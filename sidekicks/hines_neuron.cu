#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>

#include <cusp/krylov/gmres.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/bicg.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>
#include <cusp/print.h>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>



#include "gpu_timer.h"

using namespace std;

double RESTING_POTENTIAL = -60;

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


void fill_matrix_using_junction(cusp::csr_matrix<int, double, cusp::host_memory> &h_A_cusp,
								double* h_b,
								double* h_left, double* h_principal, double* h_right,
								int rows,
								vector<vector<int> > &junction_list,
								int &tridiag_nnz, int &offdiag_nnz){

	/*
	Used in moose's hsolve
	for ( i = 0; i < nCompt; i++ )
		{
			tree[ i ].Ra = 15.0 + 3.0 * i;
			tree[ i ].Rm = 45.0 + 15.0 * i;
			tree[ i ].Cm = 500.0 + 200.0 * i * i;
			Em.push_back( -0.06 );
			V.push_back( -0.06 + 0.01 * i );
		}
	*/

	tridiag_nnz = 0;
	offdiag_nnz = 0;

	int DEBUG = 0;

	if(DEBUG){
		// Printing junction list
		for (int i = 0; i < rows; ++i)
		{
			cout << i << " -> " ;
			for (int j = 0; j < junction_list[i].size(); ++j)
			{
				cout <<  junction_list[i][j] << " ";
			}
			cout << endl;
		}	
	}

	// Generate Ra's, Rm's, Cm's, Ga's
	double* Ra_ = new double[rows];
	double* Rm_ = new double[rows];
	double* Cm_ = new double[rows];
	double* Ga_ = new double[rows];
	double dt = 0.01;

	for (int i = 0; i < rows; ++i){
		//Ra_[i] = 15.0;
		//Ra_[i] = 15.0 + 3.0 * i;
		Ra_[i] = rand()%10 + 2;

		//Rm_[i] = 45.0; 
		//Rm_[i] = 45.0 + 15.0 * i;
		Rm_[i] = rand()%10 + 2;

		//Cm_[i] = 50.0; 
		//Cm_[i] =500.0 + 200.0 * i * i;
		//Cm_[i] =5.0 + 2.0 * i * i;
		Cm_[i] = rand()%10 + 2;

		//Cm_[i] = 1;
		Ga_[i] = 1/Ra_[i];
	}		

	// Passive admittance in main diagonal
	double* main_coeff = new double[rows]();

	// Initializing main diagonal with some passive components
	for (int i = 0; i < rows; ++i)
		main_coeff[i] = (Cm_[i]/dt + 1.0/Rm_[i]);

	// Non zero elements in csr format.
	vector<pair<int,double> > non_zero_elements;
	int node1,node2;
	double gi,gj,gij;
	double junction_sum;

	// Handling linear cases
	for (int i = 0; i < rows; ++i)
	{	
		if(junction_list[i].size() == 2){

			node1 = junction_list[i][0];
			node2 = junction_list[i][1];

			gi = Ga_[node1];
			gj = Ga_[node2];
			gij = gi*gj/(gi + gj);

			main_coeff[node1] += gij;
			main_coeff[node2] += gij;

			non_zero_elements.push_back(make_pair(node1*rows+node2, -1*gij));
			non_zero_elements.push_back(make_pair(node2*rows+node1, -1*gij));

			if(abs(node2-node1) == 1) // left or right principal diagonal
				tridiag_nnz += 2;
			else
				offdiag_nnz += 2;

		}
	}	

	// Handling branches
	// Generating symmetrix admittance graph
	// using junction information.
	for (int i = 0; i < rows; ++i)
	{
		if(junction_list[i].size() > 2){

			// Calculating junction sum
			junction_sum = 0;
			for (int k = 0; k < junction_list[i].size(); ++k)
				junction_sum += Ga_[junction_list[i][k]];

			// Inducing passive effect to main diagonal
			node1 = junction_list[i][0];
			main_coeff[node1] += Ga_[node1]*(1.0 - (Ga_[node1]/junction_sum) );

			// Putting admittance in off diagonal elements.
			for (int j = 0; j < junction_list[i].size(); ++j)
			{	
				node1 = junction_list[i][j];
				
				for (int k = j+1; k < junction_list[i].size(); ++k)
				{
					node2 = junction_list[i][k];

					gi = Ga_[node1];
					gj = Ga_[node2];
					gij = (gi*gj)/junction_sum;

					//cout << junction_sum << " " << gi[node1] << " " << gi[node2] << " " << admittance << endl;

					// Pushing element and its symmetry.
					non_zero_elements.push_back(make_pair(node1*rows+node2, -1*gij));
					non_zero_elements.push_back(make_pair(node2*rows+node1, -1*gij));

					if(abs(node2-node1) == 1) // left or right principal diagonal
						tridiag_nnz += 2;
					else
						offdiag_nnz += 2;

				}
			}
		}
	}


	// Add main diagonal to non_zero_elements.
	for (int i = 0; i < rows; ++i){
		tridiag_nnz += 1; // Main diagonal is part of tri diagonal
		non_zero_elements.push_back(make_pair(i*rows+i, main_coeff[i]));
	}
	

	// Initializing a cusp csr matrix
	int nnz = non_zero_elements.size();
	h_A_cusp.resize(rows, rows, nnz);

	// Getting elements in csr format.
	// and populating tri diagonal
	sort(non_zero_elements.begin(), non_zero_elements.end());

	int r,c;
	double value;
	for (int i = 0; i < nnz; ++i)
	{
		r = non_zero_elements[i].first/rows;
		c = non_zero_elements[i].first%rows;
		value = non_zero_elements[i].second;

		h_A_cusp.row_offsets[r]++;
		h_A_cusp.column_indices[i] = c;
		h_A_cusp.values[i] = value;

		if(r==c)
			h_principal[r] = value;
		if(r==c+1)
			h_left[r] = value;
		if(r==c-1)
			h_right[r] = value;
	}

	int temp;
	int sum = 0;
	// Scan operation on rowPtr;
	for (int i = 0; i < rows+1; ++i)
	{
		temp = h_A_cusp.row_offsets[i];
		h_A_cusp.row_offsets[i] = sum;
		sum += temp;
	}

	if(DEBUG)
		print_matrix(h_A_cusp);


	// Populating rhs
	for (int i = 0; i < rows; ++i){
		h_b[i] = (RESTING_POTENTIAL*Cm_[i])/dt + (RESTING_POTENTIAL/Rm_[i]);
		//h_b[i] = rand()%10 + 2;
	}

	if(DEBUG)
		for (int i = 0; i < rows; ++i)
			cout << h_b[i] << endl;


}

void generate_neural_structure(cusp::csr_matrix<int, double, cusp::host_memory> &h_A_cusp,
								double* h_b,
								double* h_left, double* h_principal, double* h_right,
								int rows, int num_mutations,
								int &tridiag_nnz, int &offdiag_nnz){

	// Juntion list with compartment numbers.
	vector< vector<int> > junction_list(rows);

	// Initializing junctions.
	for (int i = 0; i < rows; ++i)
		junction_list[i].push_back(i);
	
	
	// Marking random components for mutations.
	bool* mutations = new bool[rows];
	int mutations_found = 0;
	int mutated_comp;

	cout << num_mutations << endl;

	while(mutations_found != num_mutations){
		//cout << "here" << mutations_found << " " << num_mutations <<  endl;
		mutated_comp = (rand()%(rows-2))+2;

		if(!mutations[mutated_comp]){
			mutations_found++;
			mutations[mutated_comp] = true;
		}
	}

	// Assimilating junction information.
	int prev_node;
	for (int i = 1; i < rows; ++i)
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

	
	/*
	junction_list[0].push_back(1);
	junction_list[0].push_back(2);
	junction_list[2].push_back(3);
	junction_list[3].push_back(4);
	*/

	fill_matrix_using_junction(h_A_cusp, h_b,
								h_left, h_principal, h_right,
								rows,
								junction_list,
								tridiag_nnz, offdiag_nnz);


}

void read_neuron_structure(char* file_name, cusp::csr_matrix<int, double, cusp::host_memory> &h_A_cusp,
								double* h_b,
								double* h_left, double* h_principal, double* h_right,
								int rows, int &tridiag_nnz, int &offdiag_nnz){

	// Juntion list with compartment numbers.
	vector< vector<int> > junction_list(rows);

	// Pushing the <id compartment> of junction.
	for (int i = 0; i < rows; ++i)
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

	fill_matrix_using_junction(h_A_cusp, h_b,
								h_left, h_principal, h_right,
								rows,
								junction_list,
								tridiag_nnz, offdiag_nnz);	
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


int main(int argc, char *argv[])
{	

	bool ANALYSIS = true;
	bool FROM_FILE = true;

	int rows, num_mutations;

	int tridiag_nnz = 0;
	int offdiag_nnz = 0;

	// set stopping criteria:
	int  iteration_limit    = 100;
	double  relative_tolerance = 1e-6;

	GpuTimer tridiagTimer;
	GpuTimer cuspZeroTimer;
	GpuTimer cuspHintTimer;

	
	// cusparse handle
	cusparseHandle_t cusparse_handle = 0;
	cusparseCreate(&cusparse_handle);

	srand(time(NULL));

	FROM_FILE = atoi(argv[1]);

	if(FROM_FILE){
		rows = get_rows_from_file(argv[2]);

		if(argc > 3)
			ANALYSIS = atoi(argv[3]);

		if(argc > 4)
			iteration_limit = atoi(argv[4]);

		if(argc > 5)
			relative_tolerance = pow(10,-1*atoi(argv[5]));

	}else{
		rows = atoi(argv[2]);
		//int num_mutations = atoi(argv[2];
		num_mutations = (atof(argv[3])*(rows-2))/100;

		if(argc > 4)
			ANALYSIS = atoi(argv[4]);

		if(argc > 5)
			iteration_limit = atoi(argv[5]);

		if(argc > 6)
			relative_tolerance = pow(10,-1*atoi(argv[6]));

	}

	// Matrix details
	cusp::csr_matrix<int, double, cusp::host_memory> h_A;

	// Pointers for tri-diagonal in host and device
	double* h_left, *h_principal, *h_right;
	double* d_left, *d_principal, *d_right;

	double* h_b;
	double* d_b;

	// Allocating memory for tri diagonal and rhs
	h_left = (double*) calloc(rows, sizeof(double));
	h_principal = (double*) calloc(rows, sizeof(double));
	h_right = (double*) calloc(rows, sizeof(double));
	h_b = (double*) calloc(rows+1, sizeof(double));

	// Allocating device memory for tridiagonal and rhs
	cudaMalloc((void**)&d_left, rows*sizeof(double));
	cudaMalloc((void**)&d_principal, rows*sizeof(double));
	cudaMalloc((void**)&d_right, rows*sizeof(double));
	
	cudaMalloc((void**)& d_b, rows*sizeof(double));

	if(FROM_FILE){
		read_neuron_structure(argv[2] ,h_A, h_b ,
							  h_left, h_principal, h_right,
							  rows, tridiag_nnz, offdiag_nnz);
	}else{
		generate_neural_structure(h_A, h_b,
								h_left, h_principal, h_right,
								rows, num_mutations, tridiag_nnz, offdiag_nnz);	
	}
	

	// Copy to gpu
	cudaMemcpy(d_left, h_left, sizeof(double)*rows, cudaMemcpyHostToDevice);
	cudaMemcpy(d_principal, h_principal, sizeof(double)*rows, cudaMemcpyHostToDevice);
	cudaMemcpy(d_right, h_right, sizeof(double)*rows, cudaMemcpyHostToDevice);

	cudaMemcpy(d_b, h_b, sizeof(double)*rows, cudaMemcpyHostToDevice);

	// Generate device csr matrix
	cusp::csr_matrix<int, double, cusp::device_memory> d_A(h_A);

	// Solving Tri-diag(A)
	tridiagTimer.Start();
	cusparseDgtsv(cusparse_handle,
				  h_A.num_rows,
				  1,
				  d_left, d_principal, d_right,
				  d_b, h_A.num_rows);
	cudaDeviceSynchronize();
	tridiagTimer.Stop();

	double* h_tr_sol = (double*) calloc(h_A.num_rows, sizeof(double));
	cudaMemcpy(h_tr_sol, d_b, sizeof(double)* h_A.num_rows, cudaMemcpyDeviceToHost);

	// Solving CG using tridiag

	// allocate storage for solution (x) and right hand side (b)
	cusp::array1d<double, cusp::host_memory> cusp_h_x(h_A.num_rows);
	cusp::array1d<double, cusp::host_memory> cusp_h_b(h_A.num_rows);

	for (int i = 0; i < h_A.num_rows; ++i)
	{
		cusp_h_x[i] = h_tr_sol[i];
		cusp_h_b[i] = h_b[i];
	}

	//cusp::print(cusp_h_x);

	// allocate storage for solution (x) and right hand side (b)
	cusp::array1d<double, cusp::device_memory> cusp_d_clever_x(cusp_h_x);
	cusp::array1d<double, cusp::device_memory> cusp_d_zero_x(h_A.num_rows, 0);

	cusp::array1d<double, cusp::device_memory> cusp_d_clever_b(cusp_h_b);
	cusp::array1d<double, cusp::device_memory> cusp_d_zero_b(cusp_h_b);


	cusp::monitor<double> cleverMonitor(cusp_d_clever_b, iteration_limit, relative_tolerance, 0, !ANALYSIS);
	cusp::monitor<double> zeroMonitor(cusp_d_zero_b, iteration_limit, relative_tolerance, 0, !ANALYSIS);

	// solve the linear system A * x = b with the Conjugate Gradient method
	cuspZeroTimer.Start();
		cusp::krylov::gmres(d_A, cusp_d_zero_x, cusp_d_zero_b, iteration_limit, zeroMonitor);
		//cusp::krylov::cg(d_A, cusp_d_zero_x, cusp_d_zero_b, zeroMonitor);
		cudaDeviceSynchronize();
	cuspZeroTimer.Stop();


	cuspHintTimer.Start();
		cusp::krylov::gmres(d_A, cusp_d_clever_x, cusp_d_clever_b, iteration_limit, cleverMonitor);
		//cusp::krylov::cg(d_A, cusp_d_clever_x, cusp_d_clever_b, cleverMonitor);
		cudaDeviceSynchronize();
	cuspHintTimer.Stop();
	

	double tridiagTime = tridiagTimer.Elapsed();
	double cuspHintTime = cuspHintTimer.Elapsed();
	double cuspZeroTime = cuspZeroTimer.Elapsed();

	double clever_time = tridiagTime+cuspHintTime;
	double speedup = cuspZeroTime/clever_time;
	int clever_iterations = cleverMonitor.iteration_count();
	int bench_iterations = zeroMonitor.iteration_count();

	double offdiag_perc = ((double)offdiag_nnz*100)/h_A.num_entries;
	double tridiag_occupancy = ((double)tridiag_nnz*100)/(3*rows-2);


	if(ANALYSIS){
		string fn(argv[2]);
		if(FROM_FILE){
			int index = fn.size()-1;
			while(fn[index] != '/'){
				index--;
			}
			fn = fn.substr(index+1, fn.size()-1-index);
		}

		cout << "$" << fn << " " << rows << " " << h_A.num_entries << " " << tridiag_nnz << " " << offdiag_nnz << " " << offdiag_perc << " " << tridiag_occupancy << endl;
		cout << "#" << fn << " " << speedup << " " << clever_time << " " << cuspZeroTime << " " << clever_iterations << " " << bench_iterations << " " << tridiagTime << " " << cuspHintTime << endl;

		printf("Speedup %.2f\n",speedup);
		printf("Clever Time %.2f %d (%.2f + %.2f)\n", clever_time, clever_iterations, tridiagTime, cuspHintTime);
		printf("Bench  Time %.2f %d \n", cuspZeroTime, bench_iterations);
		printf("Number of coponenets : %d\n", rows);
		printf("Number of nonzero el : %d\n", h_A.num_entries);
		printf("Number of tridiag el : %d\n", tridiag_nnz);
		printf("Number of offdiag el : %d\n", offdiag_nnz);
		printf("Off-diagonal %%       : %.2f\n", offdiag_perc);
		printf("Tridiag-Occupancy    : %.2f\n", tridiag_occupancy);
		
	}else{
		printf("Speedup %.2f\n",speedup);
		printf("Clever Time %.2f %d (%.2f + %.2f)\n", clever_time, clever_iterations, tridiagTime, cuspHintTime);
		printf("Bench  Time %.2f %d \n", cuspZeroTime, bench_iterations);
		printf("Number of coponenets : %d\n", rows);
		printf("Number of nonzero el : %d\n", h_A.num_entries);
		printf("Number of tridiag el : %d\n", tridiag_nnz);
		printf("Number of offdiag el : %d\n", offdiag_nnz);
		printf("Off-diagonal %%       : %.2f\n", offdiag_perc);
		printf("Tridiag-Occupancy    : %.2f\n", tridiag_occupancy);
	}        
	

	return 0;
}
