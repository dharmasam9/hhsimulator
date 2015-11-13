#include "hsolve_kernels.h"
#include "gpu_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

// THRUST related
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>


int main(int argc, char const *argv[])
{
	/* initialize random seed: */
  	srand (time(NULL));

	// size of the array.
	int num_comp = 3;
	int ion_chan_per_cmp = 2;
	float dt = 1;

	int num_ion_chan = num_comp * ion_chan_per_cmp;

	// Host pointers
	float* vm;
	float* m,*h,*n;
	int* chan_comp, *chan_type;
	float *chan_current;

	// Device pointers
	float* d_vm;
	float* d_m, *d_h, *d_n;
	int* d_chan_comp, *d_chan_type;
	float *d_chan_current;

	// Allocate memory for host pointers
	vm = (float*) calloc(sizeof(float), num_comp);
	m = (float*) calloc(sizeof(float), num_comp);
	h = (float*) calloc(sizeof(float), num_comp);
	n = (float*) calloc(sizeof(float), num_comp);
	chan_comp = (int*) calloc(sizeof(int), num_ion_chan);
	chan_type = (int*) calloc(sizeof(int), num_ion_chan);
	chan_current = (float*) calloc(sizeof(float), num_ion_chan);

	
	// Allocate memory for device pointers and reset
	cudaMalloc((void**)&d_vm, sizeof(float) * num_comp);
	cudaMalloc((void**)&d_chan_type, sizeof(int) * num_ion_chan);
	cudaMalloc((void**)&d_chan_comp, sizeof(int) * num_ion_chan);
	cudaMalloc((void**)&d_chan_current, sizeof(float) * num_ion_chan);

	cudaMalloc((void**)&d_m, sizeof(float) * num_comp);
	cudaMemset(d_m,0, sizeof(float) * num_comp);
	cudaMalloc((void**)&d_n, sizeof(float) * num_comp);
	cudaMemset(d_n,0, sizeof(float) * num_comp);
	cudaMalloc((void**)&d_h, sizeof(float) * num_comp);
	cudaMemset(d_h,0, sizeof(float) * num_comp);
	

	// Initializing Vm array
	for (int i = 0; i < num_comp; ++i)
	{
		vm[i] = i + 1;
	}

	// Randomly setting ion - channel type
	for (int i = 0; i < num_ion_chan; ++i)
	{
		chan_type[i] = rand()%3;
		chan_comp[i] = i/ion_chan_per_cmp;
	}

	// Transferring initialized vm, channel and compartment info
	cudaMemcpy(d_vm, vm, sizeof(float)*num_comp , cudaMemcpyHostToDevice);
	cudaMemcpy(d_chan_comp, chan_comp, sizeof(int)*num_ion_chan , cudaMemcpyHostToDevice);
	cudaMemcpy(d_chan_type, chan_type, sizeof(int)*num_ion_chan , cudaMemcpyHostToDevice);

	
	// ************************ CYCLICAL COMPUTATION *******************************


	int num_iter = 2;

	for (int i = 0; i < num_iter; ++i)
	{

		// advanceChannels
		advance_channel_m<<<1,num_comp>>>(num_comp, d_vm, d_m, dt);
		advance_channel_n<<<1,num_comp>>>(num_comp, d_vm, d_n, dt);
		advance_channel_h<<<1,num_comp>>>(num_comp, d_vm, d_h, dt);

		// Getting changes in the channel values
		cudaMemcpy(m, d_m, sizeof(float)* num_comp, cudaMemcpyDeviceToHost);
		cudaMemcpy(n, d_n, sizeof(float)* num_comp, cudaMemcpyDeviceToHost);
		cudaMemcpy(h, d_h, sizeof(float)* num_comp, cudaMemcpyDeviceToHost);
		
		printf("Time Step %d\n", i );
		printf("m-> ");
		for (int i = 0; i < num_comp; ++i)
			printf("%f ", m[i]);
			printf("\n");

		printf("n-> ");
		for (int i = 0; i < num_comp; ++i)
			printf("%f ", n[i]);
			printf("\n");

		printf("h-> ");
		for (int i = 0; i < num_comp; ++i)
			printf("%f ", h[i]);
			printf("\n");


		calculateChannelCurrents<<<1,num_ion_chan>>>(d_vm, d_m, d_n, d_h, d_chan_comp, d_chan_type, d_chan_current, num_ion_chan);
		cudaDeviceSynchronize();

		printf("ion currents ->\n");

		// Getting changes in the channel values
		cudaMemcpy(chan_current, d_chan_current, sizeof(float)* num_ion_chan, cudaMemcpyDeviceToHost);

		float temp = 0;
		for (int i = 0; i < num_ion_chan; ++i){
			printf("%f ", chan_current[i]);
			temp += chan_current[i];
		}
			printf("\n");


		thrust::device_ptr<float> d_chan_current_thrust(d_chan_current);
		float ion_current = thrust::reduce(d_chan_current_thrust, d_chan_current_thrust + num_ion_chan);
		cudaDeviceSynchronize();
		printf("total current %f expected %f\n", ion_current, temp);
	
		// TODO
			// updateMatrix()
			// advanceCalcium()
			// advanceSynChans()

	}

}
