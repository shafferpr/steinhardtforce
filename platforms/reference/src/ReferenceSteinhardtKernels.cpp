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

#include "ReferenceSteinhardtKernels.h"
#include "SteinhardtForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace SteinhardtPlugin;
using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

void ReferenceCalcSteinhardtForceKernel::initialize(const System& system, const SteinhardtForce& force) {

    particles = force.getParticles();
    cutoffDistance=force.getCutoffDistance();
}

double ReferenceCalcSteinhardtForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    int numParticles=particles.size();

    vector<Vec3> positions(numParticles);
    vector<double> M(numParticles);
    vector<double> N(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = pos[particles[i]];
    double energy=0;

    for(int i=0; i<numParticles; i+=1){
        Vec3 positioni=positions[i];
        //for(int j=threadId.x; j<numParticles; j+= blockDim.x)
        for(int j=0; j<numParticles; j++){
            if( j!=i ){
                Vec3 positionj=positions[j];
                Vec3 rij= {positioni[0]-positionj[0], positioni[1]-positionj[1], positioni[2]-positionj[2]};
                //APPLY_PERIODIC_TO_DELTA(rij)
                double rij_norm=pow(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2],0.5);
                double switch_ij=(1-pow((rij_norm-5)/1,6))/(1-pow((rij_norm-5)/1,12));
                N[i] += switch_ij;
                for(int k=0; k<numParticles; k++){
                    if (k != i){
                        Vec3 positionk=positions[k];
                        Vec3 rik= {positioni[0]-positionk[0], positioni[1]-positionk[1], positioni[2]-positionk[2]};
                        //APPLY_PERIODIC_TO_DELTA(rik)
                        double rik_norm=pow(rik[0]*rik[0] + rik[1]*rik[1] + rik[2]*rik[2],0.5);
                        double switch_ik=(1-pow((rik_norm-5)/1,6))/(1-pow((rik_norm-5)/1,12));
                        double rdot = rij[0]*rik[0] + rij[1]*rik[1] + rij[2]*rik[2];
                        double P6=(231*pow(rdot,6.0)-315*pow(rdot,4.0)+105*pow(rdot,2.0)-5)/16;
                        M[i] += P6*switch_ik*switch_ij;
                    }
                }
            }
        }
    }

    double Q6_tot=0;
    for(int i=0; i<numParticles; i++){
      Q6_tot += pow(M[i],0.5)/N[i];
    }

    Q6_tot=Q6_tot*pow(4*3.14159/13,0.5)/numParticles;
    //Q6_tot=reduceValue(Q6_tot,temp);
    vector<Vec3> F(numParticles);

    for(int i=0; i<numParticles; i++){
        Vec3 positioni=positions[i];
        //for(int j=threadId.x; j<numParticles; j+=blockDim.x){
        for(int j=0; j<numParticles; j++){
            if( j!=i ){
                Vec3 positionj=positions[j];
                Vec3 rij= {positioni[0]-positionj[0], positioni[1]-positionj[1], positioni[2]-positionj[2]};
                //APPLY_PERIODIC_TO_DELTA(rij)
                double rij_norm=pow(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2],0.5);
                Vec3 delta_rij_norm=-rij/(2*rij_norm);
                double switch_ij_numerator=(1-pow((rij_norm-5)/1,6));
                double switch_ij_denominator=(1-pow((rij_norm-5)/1,12));
                double switch_ij=switch_ij_numerator/switch_ij_denominator;
                double delta_switch_ij=(6*pow((rij_norm-5)/1,5)*switch_ij_denominator-12*pow((rij_norm-5)/1,11)*switch_ij_numerator)/(switch_ij_denominator*switch_ij_denominator);
                for(int k=0; k<numParticles; k++){
                    if(k!=i){
                        Vec3 positionk=positions[k];
                        Vec3 rik= {positioni[0]-positionk[0], positioni[1]-positionk[1], positioni[2]-positionk[2]};
                        //APPLY_PERIODIC_TO_DELTA(rik)
                        double rik_norm=pow(rik[0]*rik[0] + rik[1]*rik[1] + rik[2]*rik[2],0.5);
                        Vec3 delta_rik_norm=-rik/(2*rik_norm);
                        double switch_ik_numerator=(1-pow((rik_norm-5)/1,6));
                        double switch_ik_denominator=(1-pow((rik_norm-5)/1,12));
                        double switch_ik=switch_ik_numerator/switch_ik_denominator;
                        double delta_switch_ik=(6*pow((rik_norm-5)/1,5)*switch_ik_denominator-12*pow((rik_norm-5)/1,11)*switch_ik_numerator)/(switch_ik_denominator*switch_ik_denominator);

                        double rdot = rij[0]*rik[0] + rij[1]*rik[1] + rij[2]*rik[2];
                        double P6=(231*pow(rdot,6.0)-315*pow(rdot,4.0)+105*pow(rdot,2.0)-5)/16;
                        double delta_P6=(1386*pow(rdot,5.0)-1260*pow(rdot,3)+210*rdot)/16;
                        F[i]=delta_rij_norm*delta_switch_ij*switch_ik*P6 + delta_rik_norm*switch_ij*delta_switch_ik*P6 + delta_rij_norm*switch_ij*switch_ik*delta_P6;//this very last term is not quite right
                        F[j]=-delta_rij_norm*delta_switch_ij*switch_ik*P6 + delta_rij_norm*switch_ij*switch_ik*delta_P6;
                        F[k]=-delta_rij_norm*switch_ij*delta_switch_ik*P6 + delta_rij_norm*switch_ij*switch_ik*delta_P6;
                    }
                }
            }
        }
    }

    for(int i=0; i<numParticles; i++){
      force[particles[i]]=F[i];
    }
    return Q6_tot;
}

void ReferenceCalcSteinhardtForceKernel::copyParametersToContext(ContextImpl& context, const SteinhardtForce& force) {

    particles = force.getParticles();


}
