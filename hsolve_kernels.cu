/**********************************************

Author: Dharma teja

***********************************************/

#include "hsolve_kernels.h"
#include <stdlib.h>
#include <stdio.h>

__global__
void advance_channel_m(int num_comp, float* vm, float* m, float dt){
	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	float potential = vm[tid];

	float alpha_m = (0.1 * (25-potential)) / (exp((25-potential)*0.1) - 1);
	float beta_m = 4 * exp(-1*potential/18);

	// printf("%f %f %f\n", potential, alpha_m, beta_m);

	m[tid] = alpha_m * dt + m[tid] * ( 1 - (alpha_m + beta_m)*dt);
}


__global__
void advance_channel_n(int num_comp, float* vm, float* n, float dt){
	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	float potential = vm[tid];

	float alpha_n = (0.01 * (10-potential)) / (exp((10-potential)*0.1) - 1);
	float beta_n = 0.125 * exp(-1*potential/80);

	//printf("%f %f %f\n", potential, alpha_n, beta_n);

	n[tid] = alpha_n * dt + n[tid] * ( 1 - (alpha_n + beta_n)*dt);
}

__global__
void advance_channel_h(int num_comp, float* vm, float* h, float dt){
	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	float potential = vm[tid];

	float alpha_h = 0.07 * exp(-1*potential/20);
	float beta_h = 1 / (exp((30-potential)*0.1) + 1);

	//printf("%f %f %f\n", potential, alpha_h, beta_h);

	h[tid] = alpha_h * dt + h[tid] * ( 1 - (alpha_h + beta_h)*dt);
}

__global__
void calculateChannelCurrents(float* d_vm, float* d_m, float* d_n, float* d_h, int* d_chan_comp, int* d_chan_type, float* d_chan_current, int num_ion_chan){
	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	int channel = d_chan_type[tid];
	int compartment = d_chan_comp[tid];

	float potential = d_vm[compartment];

	float m = d_m[compartment];
	float n = d_n[compartment];
	float h = d_h[compartment];


	switch(channel){
		case 0: // SODIUM (0.12, 115) (gna, ena)
			d_chan_current[tid] = 0.12 * pow(m,3) *h * (potential - 115);
			break;
		case 1: // POTASSIUM (0.036, -12)
			d_chan_current[tid] = 0.036 * pow(n,4) * (potential + 12);
			break;
		case 2: // CALCIUM (0.0003, 10.598)
			d_chan_current[tid] = 0.0003 * (potential - 10.598);
			break;
		default:
			break;
	}


}