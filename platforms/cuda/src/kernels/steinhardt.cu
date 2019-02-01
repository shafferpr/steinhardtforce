// This file contains kernels to compute the q6 steinhardt parameter and its gradient


/**
 * Sum a value over all threads.
 */
__device__ real reduceValue(real value, volatile real* temp) {
    const int thread = threadIdx.x;
    __syncthreads();
    temp[thread] = value;
    __syncthreads();
    for (uint step = 1; step < 32; step *= 2) {
        if (thread+step < blockDim.x && thread%(2*step) == 0)
            temp[thread] = temp[thread] + temp[thread+step];
        SYNC_WARPS
    }
    for (uint step = 32; step < blockDim.x; step *= 2) {
        if (thread+step < blockDim.x && thread%(2*step) == 0)
            temp[thread] = temp[thread] + temp[thread+step];
        __syncthreads();
    }
    return temp[0];
}

/**
 * Perform the first step of computing q6.  This is executed as a single work group.
 */
extern "C" __global__ void computeSteinhardt(int numParticles, const real4* __restrict__ posq,
         const int* __restrict__ particles, real* buffer, unsigned long long* __restrict__ forceBuffers, int paddedNumAtoms, real4 periodicBoxSize, real4 invPeriodicBoxSize,
                 real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,real* __restrict__ M, real* __restrict__ N, real* __restrict__ F) {
    extern __shared__ volatile real temp[];



    for(int i=0; i < numParticles; i++){
        M[i]=0;
        N[i]=0;

    }

    //for (int i = 0; i < numParticles; i++)
    for(int i=threadIdx.x; i<numParticles; i+=blockDim.x){
        real3 positioni=trimTo3(posq[particles[i]]);
        //for(int j=threadId.x; j<numParticles; j+= blockDim.x)
        for(int j=0; j<numParticles; j++){
            if( j!=i ){
                real3 positionj=trimTo3(posq[particles[j]]);
                real3 rij= make_real3(positioni.x-positionj.x, positioni.y-positionj.y, positioni.z-positionj.z);
                APPLY_PERIODIC_TO_DELTA(rij);
                real rij_norm=pow(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z,0.5);
                real switch_ij=(1-pow((rij_norm-CUTOFF)/1,6))/(1-pow((rij_norm-CUTOFF)/1,12));
                N[i] += switch_ij;
                for(int k=0; k<numParticles; k++){
                    if (k != i){
                        real3 positionk=trimTo3(posq[particles[k]]);
                        real3 rik= make_real3(positioni.x-positionk.x, positioni.y-positionk.y, positioni.z-positionk.z);
                        APPLY_PERIODIC_TO_DELTA(rik);
                        real rik_norm=pow(rik.x*rik.x + rik.y*rik.y + rik.z*rik.z,0.5);
                        real switch_ik=(1-pow((rik_norm-CUTOFF)/1,6))/(1-pow((rik_norm-CUTOFF)/1,12));
                        real rdot = rij.x*rik.x + rij.y*rik.y + rij.z*rik.z;
                        real P6=(231*pow(rdot,6.0)-315*pow(rdot,4.0)+105*pow(rdot,2.0)-5)/16;
                        M[i] += P6*switch_ik*switch_ij;
                    }
                }
            }
        }
    }
    real Q6_tot=0;
    for(int i=threadIdx.x; i<numParticles; i++){
        Q6_tot += pow(M[i],0.5)/N[i];
    }
    Q6_tot=Q6_tot*pow(4*3.14159/13,0.5)/numParticles;
    Q6_tot=reduceValue(Q6_tot,temp);
    //real F[numParticles][3]; //bug here?
    for(int i=0; i< numParticles; i++){
        for(int j=0; j<2; j++){
            //F[i][j]=0.0;
        }
    }

    for(int i=0; i<numParticles; i++){
        real3 positioni=trimTo3(posq[particles[i]]);
        //for(int j=threadId.x; j<numParticles; j+=blockDim.x){
        for(int j=0; j<numParticles; j++){
            if( j!=i ){
                real3 positionj=trimTo3(posq[particles[j]]);
                real3 rij= make_real3(positioni.x-positionj.x, positioni.y-positionj.y, positioni.z-positionj.z);
                APPLY_PERIODIC_TO_DELTA(rij);
                real rij_norm=pow(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z,0.5);
                real3 delta_rij_norm=-rij/(2*rij_norm);
                real switch_ij_numerator=(1-pow((rij_norm-5)/1,6));
                real switch_ij_denominator=(1-pow((rij_norm-5)/1,12));
                real switch_ij=switch_ij_numerator/switch_ij_denominator;
                real delta_switch_ij=(6*pow((rij_norm-5)/1,5)*switch_ij_denominator-12*pow((rij_norm-5)/1,11)*switch_ij_numerator)/(switch_ij_denominator*switch_ij_denominator);
                for(int k=0; k<numParticles; k++){
                    if(k!=i){
                        real3 positionk=trimTo3(posq[particles[k]]);
                        real3 rik= make_real3(positioni.x-positionk.x, positioni.y-positionk.y, positioni.z-positionk.z);
                        APPLY_PERIODIC_TO_DELTA(rik);
                        real rik_norm=pow(rik.x*rik.x + rik.y*rik.y + rik.z*rik.z,0.5);
                        real3 delta_rik_norm=-rik/(2*rik_norm);
                        real switch_ik_numerator=(1-pow((rik_norm-5)/1,6));
                        real switch_ik_denominator=(1-pow((rik_norm-5)/1,12));
                        real switch_ik=switch_ik_numerator/switch_ik_denominator;
                        real delta_switch_ik=(6*pow((rik_norm-5)/1,5)*switch_ik_denominator-12*pow((rik_norm-5)/1,11)*switch_ik_numerator)/(switch_ik_denominator*switch_ik_denominator);

                        real rdot = rij.x*rik.x + rij.y*rik.y + rij.z*rik.z;
                        real P6=(231*pow(rdot,6.0)-315*pow(rdot,4.0)+105*pow(rdot,2.0)-5)/16;
                        real delta_P6=(1386*pow(rdot,5.0)-1260*pow(rdot,3)+210*rdot)/16;
                        //F[i]=delta_switch_ij*switch_ik*P6*delta_rij_norm + switch_ij*delta_switch_ik*P6*delta_rik_norm + switch_ij*switch_ik*delta_P6*delta_rij_norm;//this very last term is not quite right
                        //F[j]=-delta_switch_ij*switch_ik*P6*delta_rij_norm + switch_ij*switch_ik*delta_P6*delta_rij_norm;
                        //F[k]=-switch_ij*delta_switch_ik*P6*delta_rik_norm + switch_ij*switch_ik*delta_P6*delta_rij_norm;
                    }
                }
            }
        }
    }




    for (int i = blockDim.x*blockIdx.x+threadIdx.x; i < numParticles; i += blockDim.x*gridDim.x) {
        int index = particles[i];
        //atomicAdd(&forceBuffers[index], static_cast<unsigned long long>((long long) (F[i][0]*0x100000000)));
        //atomicAdd(&forceBuffers[index+paddedNumAtoms], static_cast<unsigned long long>((long long) (F[i][1]*0x100000000)));
        //atomicAdd(&forceBuffers[index+2*paddedNumAtoms], static_cast<unsigned long long>((long long) (F[i][2]*0x100000000)));
    }


    // Compute the correlation matrix.

    /*real R[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    real sum = 0;
    for (int i = threadIdx.x; i < numParticles; i += blockDim.x) {
        int index = particles[i];
        real3 pos = trimTo3(posq[index]) - center;
        real3 refPos = trimTo3(referencePos[index]);
        R[0][0] += pos.x*refPos.x;
        R[0][1] += pos.x*refPos.y;
        R[0][2] += pos.x*refPos.z;
        R[1][0] += pos.y*refPos.x;
        R[1][1] += pos.y*refPos.y;
        R[1][2] += pos.y*refPos.z;
        R[2][0] += pos.z*refPos.x;
        R[2][1] += pos.z*refPos.y;
        R[2][2] += pos.z*refPos.z;
        sum += dot(pos, pos);
    }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[i][j] = reduceValue(R[i][j], temp);
    sum = reduceValue(sum, temp);

    // Copy everything into the output buffer to send back to the host.

    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                buffer[3*i+j] = R[i][j];
        buffer[9] = sum;
        buffer[10] = center.x;
        buffer[11] = center.y;
        buffer[12] = center.z;
    }*/
}

/**
 * Apply forces based on the RMSD.
 */
extern "C" __global__ void computeRMSDForces(int numParticles, int paddedNumAtoms, const real4* __restrict__ posq, const real4* __restrict__ referencePos,
         const int* __restrict__ particles, const real* buffer, unsigned long long* __restrict__ forceBuffers) {
    real3 center = make_real3(buffer[10], buffer[11], buffer[12]);
    real scale = 1 / (real) (buffer[9]*numParticles);
    for (int i = blockDim.x*blockIdx.x+threadIdx.x; i < numParticles; i += blockDim.x*gridDim.x) {
        int index = particles[i];
        real3 pos = trimTo3(posq[index]) - center;
        real3 refPos = trimTo3(referencePos[index]);
        real3 rotatedRef = make_real3(buffer[0]*refPos.x + buffer[3]*refPos.y + buffer[6]*refPos.z,
                                      buffer[1]*refPos.x + buffer[4]*refPos.y + buffer[7]*refPos.z,
                                      buffer[2]*refPos.x + buffer[5]*refPos.y + buffer[8]*refPos.z);
        real3 force = (rotatedRef-pos)*scale;
        atomicAdd(&forceBuffers[index], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[index+paddedNumAtoms], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[index+2*paddedNumAtoms], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
    }
}
