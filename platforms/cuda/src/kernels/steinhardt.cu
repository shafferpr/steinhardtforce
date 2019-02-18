// This file contains kernels to compute the steinhardt parameters and its gradient


/**
 * Perform the first step of computing q6.
 */


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
	  real rij_pow6=powf((rij_norm-CUTOFF)/1,6);
          real switch_ij=(1-rij_pow6)/(1-powf(rij_pow6,2));
	  
          sumN += switch_ij;
          for(int k=0; k<numParticles; k++){
            if (k != index){
              real3 positionk=trimTo3(posq[particles[k]]);
              real3 rik= make_real3(positioni.x-positionk.x, positioni.y-positionk.y, positioni.z-positionk.z);
              APPLY_PERIODIC_TO_DELTA(rik);
              real rik_norm=sqrtf(rik.x*rik.x + rik.y*rik.y + rik.z*rik.z);
	      if(rik_norm<1.6*CUTOFF){
	      real rik_pow6=powf((rik_norm-CUTOFF)/1,6);
              real switch_ik=(1-rik_pow6)/(1-powf(rik_pow6,2));
              real rdot = (rij.x*rik.x + rij.y*rik.y + rij.z*rik.z)/(rik_norm*rij_norm);
              //real P6=(231*powf(rdot,6.0)-315*powf(rdot,4.0)+105*powf(rdot,2.0)-5)/16;
	      real P6=legendre(rdot, 6);
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
                 real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,real* __restrict__ M, real* __restrict__ N, real* __restrict__ F, real Q_tot) {

    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    real prefactor=sqrtf(4*3.14159/13)/numParticles;
    while(i < numParticles){
    	//printf("%d\n", i);
	//printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
        real3 positioni=trimTo3(posq[particles[i]]);
	
	real3 Ficomp=make_real3(0);
	real M_prefactor=-1/(2*N[i]*sqrtf(M[i]));

        for(int j=0; j<numParticles; j++){
            if( j!=i ){
                real3 positionj=trimTo3(posq[particles[j]]);
                real3 rij= make_real3(positioni.x-positionj.x, positioni.y-positionj.y, positioni.z-positionj.z);
                APPLY_PERIODIC_TO_DELTA(rij);
                real rij_norm=sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);
		if(rij_norm < 2.0*CUTOFF){
                real3 delta_rij_norm=rij/(2*rij_norm);
		real rij_pow6=powf((rij_norm-CUTOFF)/1,6);
                real switch_ij_numerator=(1-rij_pow6);
                real switch_ij_denominator=(1-powf(rij_pow6,2));
                real switch_ij=switch_ij_numerator/switch_ij_denominator;
		real rij_pow5=powf((rij_norm-CUTOFF)/1,5);
                real delta_switch_ij=(6*rij_pow5*switch_ij_denominator-12*rij_pow5*rij_pow6*switch_ij_numerator)/(switch_ij_denominator*switch_ij_denominator);
		real3 Fjcomp=make_real3(0);
                for(int k=0; k<numParticles; k++){
                    if(k!=i){
                        real3 positionk=trimTo3(posq[particles[k]]);
                        real3 rik= make_real3(positioni.x-positionk.x, positioni.y-positionk.y, positioni.z-positionk.z);
                        APPLY_PERIODIC_TO_DELTA(rik);
			
                        real rik_norm=sqrtf(rik.x*rik.x + rik.y*rik.y + rik.z*rik.z);
			if(rik_norm < 2.0*CUTOFF){
                        real3 delta_rik_norm=rik/(2*rik_norm);
			real rik_pow6=powf((rik_norm-CUTOFF)/1,6);
                        real switch_ik_numerator=(1-rik_pow6);
                        real switch_ik_denominator=(1-powf(rik_pow6,2));
                        real switch_ik=switch_ik_numerator/switch_ik_denominator;
			real rik_pow5=powf((rik_norm-CUTOFF)/1,5);
                        real delta_switch_ik=(6*rik_pow5*switch_ik_denominator-12*rik_pow5*rik_pow6*switch_ik_numerator)/(switch_ik_denominator*switch_ik_denominator);
			
                        real rdot = rij.x*rik.x + rij.y*rik.y + rij.z*rik.z/(rij_norm*rik_norm);
                        //real P6=(231*powf(rdot,6.0)-315*powf(rdot,4.0)+105*powf(rdot,2.0)-5)/16;
			real P6=legendre(rdot,6);
                        //real delta_P6=(1386*powf(rdot,5.0)-1260*powf(rdot,3)+210*rdot)/16;
			real delta_P6=legendre_deriv(rdot,6);
			real3 delta_rijj= -2*delta_rij_norm + make_real3(2*delta_rij_norm.x*delta_rij_norm.x*delta_rik_norm.x,2*delta_rij_norm.y*delta_rij_norm.y*delta_rik_norm.y,2*delta_rij_norm.z*delta_rij_norm.z*delta_rik_norm.z)/rij_norm;
			real3 delta_rikk= -2*delta_rik_norm + make_real3(2*delta_rik_norm.x*delta_rik_norm.x*delta_rij_norm.x,2*delta_rik_norm.y*delta_rik_norm.y*delta_rij_norm.y,2*delta_rik_norm.z*delta_rik_norm.z*delta_rij_norm.z)/rik_norm;
			real3 delta_rijik = -delta_rijj-delta_rikk;
                        Ficomp += delta_switch_ij*switch_ik*P6*delta_rij_norm + switch_ij*delta_switch_ik*P6*delta_rik_norm + switch_ij*switch_ik*delta_P6*delta_rijik;
                        Fjcomp += -delta_switch_ij*switch_ik*P6*delta_rij_norm +switch_ij*switch_ik*delta_P6*delta_rijj;
                        real3 Fkcomp = -switch_ij*delta_switch_ik*P6*delta_rik_norm+switch_ij*switch_ik*delta_P6*delta_rikk;
			F[3*k]-=Fkcomp.x/M_prefactor;
			F[3*k+1]-=Fkcomp.y/M_prefactor;
			F[3*k+2]-=Fkcomp.z/M_prefactor;
                    }
                }
		}
		F[3*j]-=Fjcomp.x/M_prefactor;
		F[3*j+1]-=Fjcomp.y/M_prefactor;
		F[3*j+2]-=Fjcomp.z/M_prefactor;
		}
            }
        }
	//printf("%d %f %f %f %d %d\n",i, F[3*i], F[3*i+1], F[3*i+1], blockIdx.x, threadIdx.x);
	F[3*i]-=Ficomp.x/M_prefactor;
	F[3*i+1]-=Ficomp.y/M_prefactor;
	F[3*i+2]-=Ficomp.z/M_prefactor;
        i += blockDim.x*gridDim.x;
    }



}

extern "C" __global__ void applySteinhardtForces(int numParticles, const int* __restrict__ particles, unsigned long long* __restrict__ forceBuffers, int paddedNumAtoms, real* __restrict__ F) {

    for (int i = blockDim.x*blockIdx.x+threadIdx.x; i < numParticles; i += blockDim.x*gridDim.x) {
        int index = particles[i];
	real3 force=make_real3(F[3*i],F[3*i+1],F[3*i+2]);
        atomicAdd(&forceBuffers[index], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[index+paddedNumAtoms], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[index+2*paddedNumAtoms], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
    }

}