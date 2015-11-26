#include <iostream>
#include <stdlib.h>
#include <time.h>

#include <algorithm>

#include "cpu_hsolve_utils.cpp"


using namespace std;

int main(int argc, char const *argv[])
{

	/* initialize random seed: */
	srand (time(NULL));

	// Required arrays
	double simulation_time, dT;
	int time_steps = 0;

	int num_compt = 0;
	int num_channels = 0;
	int* h_channel_rowPtr;
	int* h_channels;
	double* h_Ek;

	double* h_V,*h_Cm, *h_Ga, *h_Rm, *h_Em;
	double* h_gate_m,*h_gate_h,*h_gate_n;
	double* h_current_inj;

	// Setting up simulation times
	simulation_time = 100;
	dT = 1;
	time_steps = simulation_time/dT;

	// Setting number of components
	num_compt = 5;

	// Allocating memory
	h_V  = new double[num_compt];
	h_Cm = new double[num_compt];
	h_Ga = new double[num_compt];
	h_Rm = new double[num_compt];
	h_Em = new double[num_compt];

	h_gate_m = new double[num_compt];
	h_gate_n = new double[num_compt];
	h_gate_h = new double[num_compt];

	h_current_inj = new double[time_steps];


	//First 25% and last 25% of currents to be set
	int temp = (25*time_steps)/100;
	for (int i = 0; i < temp; ++i){
		h_current_inj[i] = I_EXT;
		h_current_inj[time_steps-i] = I_EXT;
	}


	// Setting up channels
	h_channel_rowPtr = new int[num_compt+1];

	// Randomly assigning channels for compartment.
	for (int i = 0; i < num_compt; ++i)
		h_channel_rowPtr[i+1] = max(3,rand()%MAX_CHAN_PER_COMP) + h_channel_rowPtr[i];

	// Allocating memory for channel information
	num_channels = h_channel_rowPtr[num_compt];
	h_channels = new int[num_channels];
	h_Ek = new double[num_channels];

	// Randomly assigning channel types for chann in compartment.
	for (int i = 0; i < num_compt; ++i)
	{
		// Making sure compartment has atleast one Na,K,Cl channel
		int chan_type, na_count = 1, k_count = 1, cl_count = 1;
		for (int j = h_channel_rowPtr[i]; j < h_channel_rowPtr[i+1]-3; ++j)
		{
			chan_type = rand()%3;
			switch(chan_type){
				case 0:
					na_count++;
					break;
				case 1:
					k_count++;
					break;
				case 2:
					cl_count++;
					break;
			}
		}

		fill(h_channels + h_channel_rowPtr[i] , h_channels + h_channel_rowPtr[i] + na_count , 0);
		fill(h_channels + h_channel_rowPtr[i] + na_count , h_channels + h_channel_rowPtr[i] + na_count + k_count , 1);
		fill(h_channels + h_channel_rowPtr[i] + na_count + k_count , h_channels + h_channel_rowPtr[i+1] , 2);
	}


	populate_V(h_V, num_compt);
	populate_Cm(h_Cm, num_compt);
	populate_Ga(h_Ga, num_compt);
	populate_Rm(h_Rm, num_compt);
	populate_Em(h_Em, num_compt);

	populate_Ek(h_Ek, h_channels, num_channels);

	// ****************************** Generate Matrix ************************************




	return 0;
}