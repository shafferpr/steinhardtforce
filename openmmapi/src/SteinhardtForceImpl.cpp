/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018 Stanford University and the Authors.           *
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


#include "internal/SteinhardtForceImpl.h"
#include "SteinhardtKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <iostream>

using namespace SteinhardtPlugin;
using namespace OpenMM;
using namespace std;

SteinhardtForceImpl::SteinhardtForceImpl(const SteinhardtForce& owner) : owner(owner) {
}

SteinhardtForceImpl::~SteinhardtForceImpl() {
}

void SteinhardtForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcSteinhardtForceKernel::Name(), context);

    // Check for errors in the specification of particles.
    //const System& system = context.getSystem();

    int numParticles = context.getSystem().getNumParticles();

    //if (owner.getParticles().size() != numParticles)
    //throw OpenMMException("SteinhardtForce: Number of reference positions does not equal number of particles in the System");
    set<int> particles;
    for (int i : owner.getParticles()) {
        if (i < 0 || i >= numParticles) {
            stringstream msg;
            msg << "SteinhardtForce: Illegal particle index for SteinhardtForce: ";
            msg << i;
            throw OpenMMException(msg.str());
        }
        if (particles.find(i) != particles.end()) {
            stringstream msg;
            msg << "SteinhardtForce: Duplicated particle index for SteinhardtForce: ";
            msg << i;
            throw OpenMMException(msg.str());
        }
        particles.insert(i);

    }
    cout <<"trying to kernel\n";
    kernel.getAs<CalcSteinhardtForceKernel>().initialize(context.getSystem(), owner);
    cout <<"did i kernel\n";
}

double SteinhardtForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcSteinhardtForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> SteinhardtForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcSteinhardtForceKernel::Name());
    return names;
}

void SteinhardtForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcSteinhardtForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}
