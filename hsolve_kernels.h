

#ifndef _HSOLVE_KERNELS
#define _HSOLVE_KERNELS

#include <cuda_runtime.h>

__global__
void advance_channel_m(int num_comp, float* vm, float* m, float dt);

__global__
void advance_channel_n(int num_comp, float* vm, float* n, float dt);

__global__
void advance_channel_h(int num_comp, float* vm, float* h, float dt);

__global__
void calculateChannelCurrents(float* d_vm, float* d_m, float* d_n, float* d_h, int* d_chan_comp, int* d_chan_type, float* d_chan_current, int num_ion_chan);

#endif