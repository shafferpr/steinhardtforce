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
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"

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
};

void CudaCalcSteinhardtForceKernel::initialize(const System& system, const SteinhardtForce& force) {
    // Create data structures.

    bool useDouble = cu.getUseDoublePrecision();
    int elementSize = (useDouble ? sizeof(double) : sizeof(float));
    int numParticles = force.getParticles().size();
    if (numParticles == 0)
        numParticles = system.getNumParticles();
    referencePos.initialize(cu, system.getNumParticles(), 4*elementSize, "referencePos");
    particles.initialize<int>(cu, numParticles, "particles");
    buffer.initialize(cu, 13, elementSize, "buffer");
    recordParameters(force);
    info = new ForceInfo(force);
    cu.addForce(info);

    // Create the kernels.

    CUmodule module = cu.createModule(CudaKernelSources::vectorOps+CudaKernelSources::steinhardt);
    kernel1 = cu.getKernel(module, "computeRMSDPart1");
    kernel2 = cu.getKernel(module, "computeRMSDForces");
}

void CudaCalcSteinhardtForceKernel::recordParameters(const SteinhardtForce& force) {
    // Record the parameters and center the reference positions.

    vector<int> particleVec = force.getParticles();
    if (particleVec.size() == 0)
        for (int i = 0; i < cu.getNumAtoms(); i++)
            particleVec.push_back(i);
    vector<Vec3> centeredPositions = force.getReferencePositions();
    Vec3 center;
    for (int i : particleVec)
        center += centeredPositions[i];
    center /= particleVec.size();
    for (Vec3& p : centeredPositions)
        p -= center;

    // Upload them to the device.

    particles.upload(particleVec);
    vector<double4> pos;
    for (Vec3 p : centeredPositions)
        pos.push_back(make_double4(p[0], p[1], p[2], 0));
    referencePos.upload(pos, true);

    // Record the sum of the norms of the reference positions.

    sumNormRef = 0.0;
    for (int i : particleVec) {
        Vec3 p = centeredPositions[i];
        sumNormRef += p.dot(p);
    }
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
    void* args1[] = {&numParticles, &cu.getPosq().getDevicePointer(), &referencePos.getDevicePointer(),
            &particles.getDevicePointer(), &buffer.getDevicePointer()};
    cu.executeKernel(kernel1, args1, blockSize, blockSize, blockSize*sizeof(REAL));

    // Download the results, build the F matrix, and find the maximum eigenvalue
    // and eigenvector.

    vector<REAL> b;
    buffer.download(b);
    Array2D<double> F(4, 4);
    F[0][0] =  b[0*3+0] + b[1*3+1] + b[2*3+2];
    F[1][0] =  b[1*3+2] - b[2*3+1];
    F[2][0] =  b[2*3+0] - b[0*3+2];
    F[3][0] =  b[0*3+1] - b[1*3+0];
    F[0][1] =  b[1*3+2] - b[2*3+1];
    F[1][1] =  b[0*3+0] - b[1*3+1] - b[2*3+2];
    F[2][1] =  b[0*3+1] + b[1*3+0];
    F[3][1] =  b[0*3+2] + b[2*3+0];
    F[0][2] =  b[2*3+0] - b[0*3+2];
    F[1][2] =  b[0*3+1] + b[1*3+0];
    F[2][2] = -b[0*3+0] + b[1*3+1] - b[2*3+2];
    F[3][2] =  b[1*3+2] + b[2*3+1];
    F[0][3] =  b[0*3+1] - b[1*3+0];
    F[1][3] =  b[0*3+2] + b[2*3+0];
    F[2][3] =  b[1*3+2] + b[2*3+1];
    F[3][3] = -b[0*3+0] - b[1*3+1] + b[2*3+2];
    JAMA::Eigenvalue<double> eigen(F);
    Array1D<double> values;
    eigen.getRealEigenvalues(values);
    Array2D<double> vectors;
    eigen.getV(vectors);

    // Compute the RMSD.

    double msd = (sumNormRef+b[9]-2*values[3])/numParticles;
    if (msd < 1e-20) {
        // The particles are perfectly aligned, so all the forces should be zero.
        // Numerical error can lead to NaNs, so just return 0 now.
        return 0.0;
    }
    double rmsd = sqrt(msd);
    b[9] = rmsd;

    // Compute the rotation matrix.

    double q[] = {vectors[0][3], vectors[1][3], vectors[2][3], vectors[3][3]};
    double q00 = q[0]*q[0], q01 = q[0]*q[1], q02 = q[0]*q[2], q03 = q[0]*q[3];
    double q11 = q[1]*q[1], q12 = q[1]*q[2], q13 = q[1]*q[3];
    double q22 = q[2]*q[2], q23 = q[2]*q[3];
    double q33 = q[3]*q[3];
    b[0] = q00+q11-q22-q33;
    b[1] = 2*(q12-q03);
    b[2] = 2*(q13+q02);
    b[3] = 2*(q12+q03);
    b[4] = q00-q11+q22-q33;
    b[5] = 2*(q23-q01);
    b[6] = 2*(q13-q02);
    b[7] = 2*(q23+q01);
    b[8] = q00-q11-q22+q33;

    // Upload it to the device and invoke the kernel to apply forces.

    buffer.upload(b);
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    void* args2[] = {&numParticles, &paddedNumAtoms, &cu.getPosq().getDevicePointer(), &referencePos.getDevicePointer(),
            &particles.getDevicePointer(), &buffer.getDevicePointer(), &cu.getForce().getDevicePointer()};
    cu.executeKernel(kernel2, args2, numParticles);
    return rmsd;
}

void CudaCalcSteinhardtForceKernel::copyParametersToContext(ContextImpl& context, const SteinhardtForce& force) {
    if (referencePos.getSize() != force.getReferencePositions().size())
        throw OpenMMException("updateParametersInContext: The number of reference positions has changed");
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
