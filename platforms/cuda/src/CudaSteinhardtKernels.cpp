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
//#include "CudaKernelSources.h"
#include "CudaSteinhardtKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"
#include <set>

using namespace SteinhardtPlugin;
using namespace OpenMM;
using namespace std;

class CudaCalcSteinhardtForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const SteinhardtForce& force) : force(force) {
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
    double cutoffDistance;
};

void CudaCalcSteinhardtForceKernel::initialize(const System& system, const SteinhardtForce& force) {
    // Create data structures.

    bool useDouble = cu.getUseDoublePrecision();
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    int numParticles = force.getParticles().size();
    cutoffDistance=force.getCutoffDistance();
    if (numParticles == 0)
        numParticles = system.getNumParticles();

    particles.initialize<int>(cu, numParticles, "particles");
    //cutoffD.initialize<float>(cu,1,"cutoffD") //I'm not sure how this line should actually look
    buffer.initialize(cu, 13, elementSize, "buffer");
    recordParameters(force);
    info = new ForceInfo(force);
    cu.addForce(info);
    //cutoffD.upload(cutoffDistance);
    // Create the kernels.

    CUmodule module = cu.createModule(CudaSteinhardtKernelSources::vectorOps+CudaSteinhardtKernelSources::steinhardt);
    kernel1 = cu.getKernel(module, "computeSteinhardt");
    kernel2 = cu.getKernel(module, "computeSteinhardtForces");
}

void CudaCalcSteinhardtForceKernel::recordParameters(const SteinhardtForce& force) {
    // Record the parameters and center the reference positions.

    vector<int> particleVec = force.getParticles();
    if (particleVec.size() == 0)
        for (int i = 0; i < cu.getNumAtoms(); i++)
            particleVec.push_back(i);


    // Upload them to the device.

    particles.upload(particleVec);


}

double CudaCalcSteinhardtForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (cu.getUseDoublePrecision())
        return executeImpl<double>(context);
    return executeImpl<float>(context);
}

template <class REAL>
double CudaCalcSteinhardtForceKernel::executeImpl(ContextImpl& context) {
    // Execute the first kernel.

    int numParticles = particles.getSize();
    int blockSize = 256;




    void* args1[] = {&numParticles, &cu.getPosq().getDevicePointer(),
            &particles.getDevicePointer(), &cutoffDistance, &buffer.getDevicePointer()};
    cu.executeKernel(kernel1, args1, blockSize, blockSize, blockSize*sizeof(REAL));


    // Upload it to the device and invoke the kernel to apply forces.

    return 0;
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

    info->updateParticles();
    cu.invalidateMolecules(info);
}
