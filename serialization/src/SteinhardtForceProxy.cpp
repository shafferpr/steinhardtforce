/* -------------------------------------------------------------------------- *
 *                                OpenMMSteinhardt                                 *
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

#include "SteinhardtForceProxy.h"
#include "SteinhardtForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <sstream>

using namespace SteinhardtPlugin;
using namespace OpenMM;
using namespace std;

SteinhardtForceProxy::SteinhardtForceProxy() : SerializationProxy("SteinhardtForce") {
}

void SteinhardtForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const SteinhardtForce& force = *reinterpret_cast<const SteinhardtForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());

    SerializationNode& particlesNode = node.createChildNode("Particles");
    for (int i : force.getParticles())
       particlesNode.createChildNode("Particle").setIntProperty("index", i);

    SerializationNode& cutoffDistanceNode = node.createChildNode("CutoffDistance");
    cutoffDistanceNode.setDoubleProperty("cutoffDistance",force.getCutoffDistance());
    SerializationNode& steinhardtOrderNode = node.createChildNode("SteinhardtOrder");
    steinhardtOrderNode.setIntProperty("steinhardtOrder",force.getSteinhardtOrder());
}

void* SteinhardtForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    SteinhardtForce* force = NULL;
    try {

        vector<int> particles;
	double cutoffDistance;
  int steinhardtOrder;
        for (auto& particle : node.getChildNode("Particles").getChildren())
            particles.push_back(particle.getIntProperty("index"));
	      cutoffDistance=node.getChildNode("CutoffDistance").getDoubleProperty("cutoffDistance");
        steinhardtOrder=node.getChildNode("SteinhardtOrder").getIntProperty("steinhardtOrder");
        force = new SteinhardtForce(particles, cutoffDistance, steinhardtOrder);
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        return force;

    }
    catch (...) {
      if (force != NULL)
	  delete force;
      throw;
    }

}
