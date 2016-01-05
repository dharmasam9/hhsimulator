#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <thrust/device_vector.h>

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <time.h>

#include "gpu_timer.h"

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/krylov/gmres.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>

using namespace std;

double I_EXT = 50.0;

int CHANNEL_TYPE_NA = 0;
int CHANNEL_TYPE_K = 1;
int CHANNEL_TYPE_CL = 2;

int MAX_CHAN_PER_COMP = 15;

double gbar_Na = 120.0;
double gbar_K = 36.0;
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

__global__
void update_V(int threadCount, double* d_new_V, double* d_V){
	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	if(tid < threadCount){
		d_V[tid] = d_new_V[tid];
	}
}

__global__
void update_matrix(int threadCount, int num_comp, double* d_maindiag_passive, double* d_GkSum, 
				 	int* d_maindiag_map,
					double* d_A_cusp_values,
					double* d_tridiag_data){

	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	if(tid < threadCount){
		//printf("%d %lf\n", tid, d_GkSum[tid]);
		double temp = d_maindiag_passive[tid] + d_GkSum[tid];
		d_tridiag_data[num_comp + tid] = temp;
		d_A_cusp_values[d_maindiag_map[tid]] = temp;
	}
}

__global__
void calculate_currents(int threadCount, double* d_V, double* d_Cm, double dT,
						double* d_Em, double* d_Rm, 
						double* d_GkEkSum, double externalCurrent, 
						double* d_b){
	int tid = blockIdx.x* blockDim.x + threadIdx.x;	

	if(tid < threadCount){
		// printf("%lf %lf %lf %lf \n", (d_V[tid]*d_Cm[tid])/dT, d_Em[tid]/d_Rm[tid], d_GkEkSum[tid], externalCurrent);
		d_b[tid] = (d_V[tid]*d_Cm[tid])/dT + d_Em[tid]/d_Rm[tid] + d_GkEkSum[tid] + externalCurrent;
	}

}

__global__
void calculate_gk_gkek_sum(int threadCount, double* d_V,
						  double* d_gate_m, double* d_gate_h, double* d_gate_n, 
						  int* d_channel_counts, 
						  double* d_GkSum, double* d_GkEkSum){

	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	if(tid < threadCount){

		double temp;
		double gksum = 0;
		double gkeksum = 0;

		// printf("%lf %lf %lf %d\n", d_gate_m[tid], d_gate_h[tid], d_gate_n[tid] , d_channel_counts[3*tid+1]);

		temp = 120.0 * pow(d_gate_m[tid],3) * d_gate_h[tid] * d_channel_counts[3*tid];
		gksum += temp;
		gkeksum += temp*(115);

		temp = 36.0 * pow(d_gate_n[tid],4) * d_channel_counts[3*tid+1];
		gksum += temp;
		gkeksum += temp*(-12);

		temp = 0.3* d_channel_counts[3*tid+2];
		gksum += temp;
		gkeksum += temp*(10.6);

		d_GkSum[tid] = gksum;
		d_GkEkSum[tid] = gkeksum;
	}
}


__global__
void advance_channel_m(int threadCount, double* vm, double* m, double dt){
	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	if(tid < threadCount){
		double potential = vm[tid];

		double alpha_m = (0.1 * (25-potential)) / (exp((25-potential)*0.1) - 1);
		double beta_m = 4 * exp(-1*potential/18);

		m[tid] = alpha_m * dt + m[tid] * ( 1 - (alpha_m + beta_m)*dt);

		//printf("%lf %lf %lf %lf\n", potential, alpha_m, beta_m, m[tid]);
	}
}


__global__
void advance_channel_n(int threadCount, double* vm, double* n, double dt){
	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	if(tid < threadCount){
		double potential = vm[tid];

		double alpha_n = (0.01 * (10-potential)) / (exp((10-potential)*0.1) - 1);
		double beta_n = 0.125 * exp(-1*potential/80);

		n[tid] = alpha_n * dt + n[tid] * ( 1 - (alpha_n + beta_n)*dt);

		//printf("%lf %lf %lf %lf\n", potential, alpha_n, beta_n, n[tid]);
	}
}

__global__
void advance_channel_h(int threadCount, double* vm, double* h, double dt){
	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	if(tid < threadCount){

		double potential = vm[tid];

		double alpha_h = 0.07 * exp(-1*potential/20);
		double beta_h = 1 / (exp((30-potential)*0.1) + 1);

		h[tid] = alpha_h * dt + h[tid] * ( 1 - (alpha_h + beta_h)*dt);

		//printf("%lf %lf %lf %lf\n", potential, alpha_h, beta_h, h[tid]);
	}
}


//**************************************** CPU Functions ******************************

void print_matrix(cusp::csr_matrix<int, double, cusp::host_memory> &h_A_cusp){
	for (int i = 0; i < h_A_cusp.num_rows; ++i)
	{
		int j = 0;
		int k = h_A_cusp.row_offsets[i];
		// printf("%2d-> ", i);
		while(j < h_A_cusp.num_rows && k < h_A_cusp.row_offsets[i+1]){
			if(h_A_cusp.column_indices[k] == j){
				printf("%7.2f ", h_A_cusp.values[k]);
				k++;
			}else{
				printf("%7.2f ", 0);
			}
			j++;
		}

		while(j < h_A_cusp.num_rows){
			printf("%7.2f ", 0);
			j++;
		}

		cout << endl;
	}
	cout << endl;

}

void fill_matrix_using_junction(int num_comp, const vector<vector<int> > &junction_list,
								cusp::csr_matrix<int, double, cusp::host_memory> &h_A_cusp, double* h_b,
								double* h_maindiag_passive, double* h_tridiag_data, int* h_maindiag_map,
								double* h_Cm, double* h_Ga, double* h_Rm, double dT,
								int &tridiag_nnz, int &offdiag_nnz){


	int DEBUG = 0;

	tridiag_nnz = 0;
	offdiag_nnz = 0;

	// Membrance resitance and capacitance terms.
	for (int i = 0; i < num_comp; ++i)
		h_maindiag_passive[i] += (h_Cm[i]/dT + 1.0/h_Rm[i]);


	// Non zero elements in csr format.
	vector<pair<int,double> > non_zero_elements;
	int node1,node2;
	double gi,gj,gij;
	double junction_sum;

	// Handling linear cases
	for (int i = 0; i < num_comp; ++i)
	{	
		if(junction_list[i].size() == 2){

			node1 = junction_list[i][0];
			node2 = junction_list[i][1];

			gi = h_Ga[node1];
			gj = h_Ga[node2];
			gij = (gi*gj)/(gi + gj);

			h_maindiag_passive[node1] += gij;
			h_maindiag_passive[node2] += gij;

			non_zero_elements.push_back(make_pair(node1*num_comp+node2, -1*gij));
			non_zero_elements.push_back(make_pair(node2*num_comp+node1, -1*gij));

		}
	}	

	// Generating symmetrix admittance graph
	// using junction information.
	for (int i = 0; i < num_comp; ++i)
	{
		if(junction_list[i].size() > 2){

			// Calculating junction sum
			junction_sum = 0;
			for (int k = 0; k < junction_list[i].size(); ++k)
				junction_sum += h_Ga[junction_list[i][k]];

			// Inducing passive effect to main diagonal
			node1 = junction_list[i][0];
			h_maindiag_passive[node1] += h_Ga[node1]*(1.0 - h_Ga[node1]/junction_sum);

			// Putting admittance in off diagonal elements.
			for (int j = 0; j < junction_list[i].size(); ++j)
			{	
				node1 = junction_list[i][j];
				
				for (int k = j+1; k < junction_list[i].size(); ++k)
				{
					node2 = junction_list[i][k];

					gi = h_Ga[node1];
					gj = h_Ga[node2];
					gij = (gi*gj)/junction_sum;

					//cout << junction_sum << " " << gi[node1] << " " << gi[node2] << " " << admittance << endl;

					// Pushing element and its symmetry.
					non_zero_elements.push_back(make_pair(node1*num_comp+node2, -1*gij));
					non_zero_elements.push_back(make_pair(node2*num_comp+node1, -1*gij));

				}
			}
		}
	}


	// Add main diagonal to non_zero_elements.
	for (int i = 0; i < num_comp; ++i)
		non_zero_elements.push_back(make_pair(i*num_comp+i, h_maindiag_passive[i]));
	

	// Initializing a cusp csr matrix
	int nnz = non_zero_elements.size();
	h_A_cusp.resize(num_comp, num_comp, nnz);

	// Getting elements in csr format.
	// and populating tri diagonal
	sort(non_zero_elements.begin(), non_zero_elements.end());

	int r,c;
	double value;
	for (int i = 0; i < nnz; ++i)
	{
		r = non_zero_elements[i].first/num_comp;
		c = non_zero_elements[i].first%num_comp;
		value = non_zero_elements[i].second;

		h_A_cusp.row_offsets[r]++;
		h_A_cusp.column_indices[i] = c;
		h_A_cusp.values[i] = value;

		if(r==c){
			h_tridiag_data[1*num_comp + r] = value;
			h_maindiag_map[r] = i; // Maintaing csr indices of main diagonal
		}
			
		if(r==c+1)
			h_tridiag_data[r] = value;
	
		if(r==c-1)
			h_tridiag_data[2*num_comp + r] = value;
	}

	int temp;
	int sum = 0;
	// Scan operation on rowPtr;
	for (int i = 0; i < num_comp+1; ++i)
	{
		temp = h_A_cusp.row_offsets[i];
		h_A_cusp.row_offsets[i] = sum;
		sum += temp;
	}

	if(DEBUG)
		print_matrix(h_A_cusp);


	// Populating rhs
	for (int i = 0; i < num_comp; ++i)
		//h_b[i] = (rand()%20)+2;
		h_b[i] = 1;

}


int get_rows_from_file(char* file_name){
	ifstream input_file(file_name);
	string line;

	int rows = 0;
	while(getline(input_file, line)){
		if(line[0] != '#')
			rows++;
	}

	input_file.ignore();
	input_file.close();


	return rows;

}

void get_structure_from_neuron(char* file_name, int num_comp, vector< vector<int> > &junction_list){

	// Initializing junctions.
	for (int i = 0; i < num_comp; ++i)
		junction_list[i].push_back(i);
	
	// read from file and get junction data.
	ifstream input_file(file_name);
	string line;

	int start,end;
	int parent,child;
	double temp_double;
	while(getline(input_file, line)){
		if(line[0] != '#'){
			stringstream ss(line);
			ss >> child;
			ss >> parent;
			ss >> temp_double;
			ss >> temp_double;
			ss >> temp_double;
			ss >> temp_double;
			ss >> parent;

			if(parent != -1){
				junction_list[parent-1].push_back(child-1);
			}

			//cout << child << " " << parent << endl;
		}
	}

	input_file.close();
}


void generate_random_neuron(int num_comp, int num_mutations, vector<vector<int> > &junction_list){
	// Initializing junctions.
	for (int i = 0; i < num_comp; ++i)
		junction_list[i].push_back(i);
	
	// Marking random components for mutations.
	bool* mutations = new bool[num_comp];
	int mutations_found = 0;
	int mutated_comp;

	while(mutations_found != num_mutations){
		mutated_comp = (rand()%(num_comp-2))+2;

		if(!mutations[mutated_comp]){
			mutations_found++;
			mutations[mutated_comp] = true;
		}
	}

	// Assimilating junction information.
	int prev_node;
	for (int i = 1; i < num_comp; ++i)
	{
		if(!mutations[i]){
			junction_list[i-1].push_back(i);
		}else{
			// Find the component to associate 
			prev_node = rand()%(i-1);
			//cout << "(" << prev_node << "," << i << ")" << endl;
			junction_list[prev_node].push_back(i);
		}
	}


}


void populate_V(double* h_V, int num_comp){
	for (int i = 0; i < num_comp; ++i)
		h_V[i] = E_RESTING_POTENTIAL;
		//h_V[i] = 0;
}

void populate_Cm(double* h_Cm, int num_comp){
	for (int i = 0; i < num_comp; ++i)
		//h_Cm[i] = 5;
		h_Cm[i] = rand()%10 + 2.0;

}

void populate_Ga(double* h_Ga, int num_comp){
	for (int i = 0; i < num_comp; ++i)
		//h_Ga[i] = 0.2;
		h_Ga[i] = 1/(rand()%10 + 2.0);
}

void populate_Rm(double* h_Rm, int num_comp){
	for (int i = 0; i < num_comp; ++i)
		//h_Rm[i] = 5;
		h_Rm[i] = rand()%10 + 2;
}

void populate_Em(double* h_Em, int num_comp){
for (int i = 0; i < num_comp; ++i)
		h_Em[i] = E_RESTING_POTENTIAL;
}
