/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaSteinhardtKernels.h"
#include "CudaSteinhardtKernelSources.h"
#include "SteinhardtForce.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <set>
#include <iostream>
#include <cmath>

using namespace SteinhardtPlugin;
using namespace OpenMM;
using namespace std;


class CudaSteinhardtForceInfo : public CudaForceInfo {
public:
    CudaSteinhardtForceInfo(const SteinhardtForce& force) : force(force) {
      updateParticles();

    }
    void updateParticles() {
        particles.clear();
        for (int i : force.getParticles())
            particles.insert(i);
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        bool include1 = (particles.find(particle1) != particles.end());
        bool include2 = (particles.find(particle2) != particles.end());
        return (include1 == include2);
    }
private:
    const SteinhardtForce& force;
    set<int> particles;
    float cutoffDistance;
    int steinhardtOrder;
};

CudaCalcSteinhardtForceKernel::~CudaCalcSteinhardtForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CudaCalcSteinhardtForceKernel::initialize(const System& system, const SteinhardtForce& force) {
    // Create data structures.
  cu.setAsCurrent();
  int numContexts=cu.getPlatformData().contexts.size();

    bool useDouble = cu.getUseDoublePrecision();
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    int numParticles = force.getParticles().size();
    float cutoffDistance=force.getCutoffDistance();
    int steinhardtOrder=force.getSteinhardtOrder();
    map<string, string> replacements;

    if (numParticles == 0)
        numParticles = system.getNumParticles();

    particles.initialize<int>(cu, numParticles, "particles");
    M.initialize(cu,numParticles,elementSize,"M");
    N.initialize(cu,numParticles,elementSize,"N");
    F.initialize(cu,3*numParticles,elementSize,"F");

    replacements["CUTOFF"]=cu.doubleToString(cutoffDistance);
    replacements["STEINHARDT_ORDER"]=cu.intToString(steinhardtOrder);

    buffer.initialize(cu, 13, elementSize, "buffer");
    recordParameters(force);

    cu.addForce(new CudaSteinhardtForceInfo(force));



    CUmodule module = cu.createModule(CudaSteinhardtKernelSources::vectorOps+CudaSteinhardtKernelSources::steinhardt,replacements);

    kernel1 = cu.getKernel(module, "computeSteinhardt");
    kernel2 = cu.getKernel(module, "computeSteinhardtForces");
    kernel3 = cu.getKernel(module, "applySteinhardtForces");

}

void CudaCalcSteinhardtForceKernel::recordParameters(const SteinhardtForce& force) {
    int numParticles = force.getParticles().size();
    int steinhardtOrder=force.getSteinhardtOrder();
    vector<int> particleVec = force.getParticles();
    
    if (particleVec.size() == 0)
        for (int i = 0; i < cu.getNumAtoms(); i++)
            particleVec.push_back(i);
    particles.upload(particleVec);

    vector<float> Mvec;
    for (int i=0; i < numParticles; i++){
      Mvec.push_back(0.0);
    }
    M.upload(Mvec);
    N.upload(Mvec);
    vector<float> Fvec;
    for(int i=0; i< numParticles*3; i++){
      Fvec.push_back(0.0);
    }
    F.upload(Fvec);
    vector<float> bvec;
    for(int i=0; i<13; i++){
      bvec.push_back(0.0);
    }
    buffer.upload(bvec);
    // Upload them to the device.


}

double CudaCalcSteinhardtForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

  //if (cu.getUseDoublePrecision())
  //return executeImpl<double>(context);
    return executeImpl<float>(context);
}

template <class REAL>
double CudaCalcSteinhardtForceKernel::executeImpl(ContextImpl& context) {
    // Execute the first kernel.

    int numParticles = particles.getSize();
    int blockSize = 256;

    
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    void* args1[] = {&numParticles, &cu.getPosq().getDevicePointer(),
            &particles.getDevicePointer(), &buffer.getDevicePointer(), &cu.getForce().getDevicePointer(), &paddedNumAtoms,
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
		     cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(), &M.getDevicePointer(), &N.getDevicePointer(), &F.getDevicePointer()};

    cu.executeKernel(kernel1, args1, blockSize, blockSize, blockSize*sizeof(REAL));
    vector<REAL> Mvec;
    M.download(Mvec);
    vector<REAL> Nvec;
    N.download(Nvec);
    
    REAL Q_tot=0;
    for(int i=0; i<numParticles; i++){
        Q_tot += pow(Mvec[i],0.5)/Nvec[i];
    }
    Q_tot=Q_tot*pow(4*3.14159/(2*steinhardtOrder+1),0.5)/numParticles;
    
    void* args2[] = {&numParticles, &cu.getPosq().getDevicePointer(),
            &particles.getDevicePointer(), &buffer.getDevicePointer(), &cu.getForce().getDevicePointer(), &paddedNumAtoms,
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(),
		     cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(), &M.getDevicePointer(), &N.getDevicePointer(), &F.getDevicePointer(), &Q_tot};


    cu.executeKernel(kernel2, args2, blockSize, blockSize, blockSize*sizeof(REAL));

    void* args3[] = {&numParticles, &particles.getDevicePointer(), &cu.getForce().getDevicePointer(), &paddedNumAtoms, &F.getDevicePointer()};

    cu.executeKernel(kernel3, args3, blockSize, blockSize, blockSize*sizeof(REAL));
    return Q_tot;
}

void CudaCalcSteinhardtForceKernel::copyParametersToContext(ContextImpl& context, const SteinhardtForce& force) {
  /*if (referencePos.getSize() != force.getReferencePositions().size())
    throw OpenMMException("updateParametersInContext: The number of reference positions has changed");*/

    int numParticles = force.getParticles().size();
    if (numParticles == 0)
        numParticles = context.getSystem().getNumParticles();
    if (numParticles != particles.getSize())
        particles.resize(numParticles);

    recordParameters(force);

    // Mark that the current reordering may be invalid.

    //info->updateParticles();
    cu.invalidateMolecules();
}
