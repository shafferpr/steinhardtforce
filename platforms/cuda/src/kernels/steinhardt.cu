// This file contains kernels to compute the steinhardt parameters and its gradient


/**
 * Perform the first step of computing q6.
 */

 __device__ real reduceValue(real value, volatile real* temp) {
     const int thread = threadIdx.x;
     __syncthreads();
     temp[thread] = value;
     __syncthreads();
     for (unsigned int step = 1; step < 32; step *= 2) {
         if (thread+step < blockDim.x && thread%(2*step) == 0)
             temp[thread] = temp[thread] + temp[thread+step];
         SYNC_WARPS
     }
     for (unsigned int step = 32; step < blockDim.x; step *= 2) {
         if (thread+step < blockDim.x && thread%(2*step) == 0)
             temp[thread] = temp[thread] + temp[thread+step];
         __syncthreads();
     }
     return temp[0];
 }


__device__ real legendre(real rdot,int steinhardt_order){
	 real result=0;

	 if(steinhardt_order == 6)
	 	result= (231*powf(rdot,6.0)-315*powf(rdot,4.0)+105*powf(rdot,2.0)-5)/16;
	 else if (steinhardt_order == 4)
	 	result=(35*powf(rdot,4.0)-30*powf(rdot,2.0)+3)/8;

	 return result;
}

__device__ real legendre_deriv(real rdot, int steinhardt_order){
	 real result=0;
	 if(steinhardt_order==6)
	 	result=(1386*powf(rdot,5.0)-1260*powf(rdot,3)+210*rdot)/16;
	 else if(steinhardt_order == 4)
	 	result=(140*powf(rdot,3.0)-60*rdot)/8;

	 return result;
}

extern "C" __global__ void computeSteinhardt(int numParticles, const real4* __restrict__ posq,
         const int* __restrict__ particles, real* buffer, unsigned long long* __restrict__ forceBuffers, int paddedNumAtoms, real4 periodicBoxSize, real4 invPeriodicBoxSize,
                 real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,real* __restrict__ M, real* __restrict__ N, real* __restrict__ F) {
    extern __shared__ volatile real temp[];


		unsigned int index = blockIdx.x*blockDim.x+threadIdx.x;
		while(index < numParticles){
			real3 positioni=trimTo3(posq[particles[index]]);
			real sumN=0;
			real sumM=0;
			F[3*index]=0;
			F[3*index+1]=0;
			F[3*index+2]=0;
			for(int j=0; j<numParticles; j++){
				if( j!=index ){
					real3 positionj=trimTo3(posq[particles[j]]);
					real3 rij= make_real3(positioni.x-positionj.x, positioni.y-positionj.y, positioni.z-positionj.z);
					APPLY_PERIODIC_TO_DELTA(rij);
					real rij_norm=sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);
					if(rij_norm<1.6*CUTOFF){
						//real rij_pow6=powf((rij_norm-CUTOFF)/1,6);
						//real switch_ij=(1-rij_pow6)/(1-powf(rij_pow6,2));
						real switch_ij=(1-tanhf((rij_norm-CUTOFF)/0.2));
						sumN += switch_ij;
						for(int k=0; k<numParticles; k++){
							if (k != index){
								real3 positionk=trimTo3(posq[particles[k]]);
								real3 rik= make_real3(positioni.x-positionk.x, positioni.y-positionk.y, positioni.z-positionk.z);
								APPLY_PERIODIC_TO_DELTA(rik);
								real rik_norm=sqrtf(rik.x*rik.x + rik.y*rik.y + rik.z*rik.z);
								if(rik_norm<1.6*CUTOFF){
									//real rik_pow6=powf((rik_norm-CUTOFF)/1,6);
									//real switch_ik=(1-rik_pow6)/(1-powf(rik_pow6,2));

									real switch_ik=(1-tanhf((rik_norm-CUTOFF)/0.2));
									real rdot = (rij.x*rik.x + rij.y*rik.y + rij.z*rik.z)/(rik_norm*rij_norm);
									//real P6=(231*powf(rdot,6.0)-315*powf(rdot,4.0)+105*powf(rdot,2.0)-5)/16;
									real P6=legendre(rdot, STEINHARDT_ORDER);
									//M[i] += P6*switch_ik*switch_ij;
									sumM += P6*switch_ik*switch_ij;
								}
							}
						}
					}
				}
			}
			M[index]=sumM;
			N[index]=sumN;
			index += blockDim.x*gridDim.x;
		}


}

/**
 * Compute forces by calculating the derivative
*/
extern "C" __global__ void computeSteinhardtForces(int numParticles, const real4* __restrict__ posq,
	const int* __restrict__ particles, real* buffer, unsigned long long* __restrict__ forceBuffers, int paddedNumAtoms, real4 periodicBoxSize, real4 invPeriodicBoxSize,
	real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,real* __restrict__ M, real* __restrict__ N, real* F, real Q_tot) {
		extern __shared__ volatile real temp[];
		unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
		real prefactor=sqrtf(4*3.14159/13)/numParticles;


		while(i < numParticles){

			real3 positioni=trimTo3(posq[particles[i]]);

			real3 Ficomp=make_real3(0);
      real3 Ficomp2=make_real3(0);
			real M_prefactor=sqrtf(4*3.14159/(13*M[i]))/(2*numParticles*N[i]);
			real N_prefactor=-sqrtf(M[i]*4*3.14159/13)/(N[i]*N[i]*numParticles);

			for(int j=0; j<numParticles; j++){
        real3 Fjcomp2=make_real3(0);
				if( j!=i ){
					real3 positionj=trimTo3(posq[particles[j]]);
					real3 rij= make_real3(positioni.x-positionj.x, positioni.y-positionj.y, positioni.z-positionj.z);
					APPLY_PERIODIC_TO_DELTA(rij);
					real rij_norm=sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);
					if(rij_norm < 2.0*CUTOFF){

						real3 delta_rij_norm=rij/(2*rij_norm);

						real sech_ij=1/coshf((rij_norm-CUTOFF)/0.2);
						real switch_ij = (1-tanhf((rij_norm-CUTOFF)/0.2));
						real delta_switch_ij = -5*powf(sech_ij,2.0);

						real3 Fjcomp=make_real3(0);
            Fjcomp2 = -delta_switch_ij*delta_rij_norm;
            Ficomp2 += delta_switch_ij*delta_rij_norm;

						for(int k=0; k<numParticles; k++){
							if(k!=i){
								real3 positionk=trimTo3(posq[particles[k]]);
								real3 rik= make_real3(positioni.x-positionk.x, positioni.y-positionk.y, positioni.z-positionk.z);
								APPLY_PERIODIC_TO_DELTA(rik);

								real rik_norm=sqrtf(rik.x*rik.x + rik.y*rik.y + rik.z*rik.z);
								if(rik_norm < 2.0*CUTOFF){
									real3 delta_rik_norm=rik/(2*rik_norm);

									real switch_ik = (1-tanhf((rik_norm-CUTOFF)/0.2));
									real sech_ik=1/coshf((rik_norm-CUTOFF)/0.2);
									real delta_switch_ik = -5*powf(sech_ik,2.0);

									real rdot = (rij.x*rik.x + rij.y*rik.y + rij.z*rik.z)/(rij_norm*rik_norm);

									real P6=legendre(rdot,STEINHARDT_ORDER);

									real delta_P6=legendre_deriv(rdot,STEINHARDT_ORDER);

                  real3 delta_rijj=-delta_rik_norm/rij_norm + delta_rij_norm*rdot/rij_norm;
                  real3 delta_rikk=-delta_rij_norm/rik_norm + delta_rik_norm*rdot/rik_norm;
                  real3 delta_rijik = -delta_rijj - delta_rikk;
									Ficomp += delta_switch_ij*switch_ik*P6*delta_rij_norm + switch_ij*delta_switch_ik*P6*delta_rik_norm + switch_ij*switch_ik*delta_P6*delta_rijik;
									Fjcomp += -delta_switch_ij*switch_ik*P6*delta_rij_norm +switch_ij*switch_ik*delta_P6*delta_rijj;
									real3 Fkcomp = -switch_ij*delta_switch_ik*P6*delta_rik_norm + switch_ij*switch_ik*delta_P6*delta_rikk;

									atomicAdd(&F[3*k],-Fkcomp.x/M_prefactor);
									atomicAdd(&F[3*k+1],-Fkcomp.y/M_prefactor);
									atomicAdd(&F[3*k+2],-Fkcomp.z/M_prefactor);

								}
							}
						}

						atomicAdd(&F[3*j],-Fjcomp.x/M_prefactor-Fjcomp2.x/N_prefactor);
						atomicAdd(&F[3*j+1],-Fjcomp.y/M_prefactor-Fjcomp2.y/N_prefactor);
						atomicAdd(&F[3*j+2],-Fjcomp.z/M_prefactor-Fjcomp2.z/N_prefactor);
					}
				}
			}

			atomicAdd(&F[3*i],-Ficomp.x/M_prefactor-Ficomp2.x/N_prefactor);
			atomicAdd(&F[3*i+1],-Ficomp.y/M_prefactor-Ficomp2.y/N_prefactor);
			atomicAdd(&F[3*i+2],-Ficomp.z/M_prefactor-Ficomp2.z/N_prefactor);
			i += blockDim.x*gridDim.x;

		}



	}

extern "C" __global__ void applySteinhardtForces(int numParticles, const int* __restrict__ particles, unsigned long long* __restrict__ forceBuffers, int paddedNumAtoms, real* __restrict__ F) {

    for (int i = blockDim.x*blockIdx.x+threadIdx.x; i < numParticles; i += blockDim.x*gridDim.x) {
        int index = particles[i];
				real3 force=make_real3(F[3*i],F[3*i+1],F[3*i+2]);
				if(i==0){
					printf("hi %f %f %f\n", F[3*i], F[3*i+1], F[3*i+2]);
				}
        atomicAdd(&forceBuffers[index], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[index+paddedNumAtoms], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[index+2*paddedNumAtoms], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
    }

}
