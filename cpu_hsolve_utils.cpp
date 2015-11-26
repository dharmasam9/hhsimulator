

double I_EXT = 50.0;

int CHANNEL_TYPE_NA = 0;
int CHANNEL_TYPE_K = 1;
int CHANNEL_TYPE_CL = 2;

int MAX_CHAN_PER_COMP = 6;

double gbar_K = 36.0;
double gbar_Na = 120.0;
double gbar_L = 0.3;

double E_RESTING_POTENTIAL = -70;
double E_NA = 115.0;
double E_K = -12.0;
double E_CL = 10.6;

/* 
// Populating method using in MOOSE
tree[ i ].Ra = 15.0 + 3.0 * i;
tree[ i ].Rm = 45.0 + 15.0 * i;
tree[ i ].Cm = 500.0 + 200.0 * i * i;
Em.push_back( -0.06 );
V.push_back( -0.06 + 0.01 * i );
*/



void populate_V(double* h_V, int num_compt){
	for (int i = 0; i < num_compt; ++i)
		h_V[i] = E_RESTING_POTENTIAL;
}

void populate_Cm(double* h_Cm, int num_compt){
	for (int i = 0; i < num_compt; ++i)
		h_Cm[i] = 1.0;

}

void populate_Ga(double* h_Ga, int num_compt){
	for (int i = 0; i < num_compt; ++i)
		h_Ga[i] = 2;
}

void populate_Rm(double* h_Rm, int num_compt){
	for (int i = 0; i < num_compt; ++i)
		h_Rm[i] = 2;
}

void populate_Em(double* h_Em, int num_compt){
for (int i = 0; i < num_compt; ++i)
		h_Em[i] = E_RESTING_POTENTIAL;
}

void populate_Ek(double* h_Ek, int* h_channels, int num_channels){
	for (int i = 0; i < num_channels; ++i)
	{
		if(h_channels[i] == CHANNEL_TYPE_NA){
			h_Ek[i] = E_NA;
			continue;
		}

		if(h_channels[i] == CHANNEL_TYPE_K){
			h_Ek[i] = E_K;
			continue;
		}

		if(h_channels[i] == CHANNEL_TYPE_CL){
			h_Ek[i] = E_CL;
			continue;
		}
	}
}
