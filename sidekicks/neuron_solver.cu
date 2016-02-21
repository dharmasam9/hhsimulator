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

void print_matrix(cusp::csr_matrix<int, float, cusp::host_memory> &h_A_cusp){
	for (int i = 0; i < h_A_cusp.num_rows; ++i)
	{
		int j = 0;
		int k = h_A_cusp.row_offsets[i];
		// printf("%2d-> ", i);
		while(j < h_A_cusp.num_rows && k < h_A_cusp.row_offsets[i+1]){
			if(h_A_cusp.column_indices[k] == j){
				if(h_A_cusp.values[k] < 10)
					cout << h_A_cusp.values[k] << "  ";
				else
					cout << h_A_cusp.values[k] << " ";

				k++;
			}else{
				cout << "0" << "  ";
			}

			j++;
		}

		while(j < h_A_cusp.num_rows){
			cout << "0" << "  ";
			j++;
		}

		cout << endl;
	}
	cout << endl;
}


void fill_matrix_using_junction(cusp::csr_matrix<int, float, cusp::host_memory> &h_A_cusp,
								float* h_b,
								float* h_left, float* h_principal, float* h_right,
								int rows,
								float* gi, vector<vector<int> > &junction_list, float* junction_sums,
								int &tridiag_nnz, int &offdiag_nnz){

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
	

	// Generating symmetrix admittance graph
	// using junction information.
	vector<pair<int,float> > non_zero_elements;
	int node1,node2;
	float admittance;
	float junction_sum;

	for (int i = 0; i < rows; ++i)
	{
		// Main diagonal element
		admittance = gi[i];
		non_zero_elements.push_back(make_pair(i*rows+i, admittance));

		tridiag_nnz++; // [i,i] is part of tri diagonal

		junction_sum = junction_sums[i];
		for (int j = 0; j < junction_list[i].size(); ++j)
		{	
			node1 = junction_list[i][j];
			for (int k = j+1; k < junction_list[i].size(); ++k)
			{
				node2 = junction_list[i][k];
				admittance = (float)(gi[node1]*gi[node2])/junction_sum;

				
				//cout << junction_sum << " " << gi[node1] << " " << gi[node2] << " " << admittance << endl;

				// Pushing element and its symmetry.
				non_zero_elements.push_back(make_pair(node1*rows+node2, admittance));
				non_zero_elements.push_back(make_pair(node2*rows+node1, admittance));

				if(abs(node2-node1) == 1) // left or right principal diagonal
					tridiag_nnz += 2;
				else
					offdiag_nnz += 2;
			}
		}
	}


	// Initializing a cusp csr matrix
	int nnz = non_zero_elements.size();
	h_A_cusp.resize(rows, rows, nnz);

	// Getting elements in csr format.
	// and populating tri diagonal
	sort(non_zero_elements.begin(), non_zero_elements.end());

	int r,c,value;
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

	//cusp::print(h_A_cusp);
	if(DEBUG)
		print_matrix(h_A_cusp);


	// Populating rhs
	for (int i = 0; i < rows; ++i)
		//h_b[i] = (rand()%20)+2;
		h_b[i] = 1;

}

void generate_neural_structure(cusp::csr_matrix<int, float, cusp::host_memory> &h_A_cusp,
								float* h_b,
								float* h_left, float* h_principal, float* h_right,
								int rows, int num_mutations,
								int &tridiag_nnz, int &offdiag_nnz){

	// Juntion list with compartment numbers.
	vector< vector<int> > junction_list(rows);
	float rand_number;

	float* gi = new float[rows];
	float* junction_sums = new float[rows];

	// Generating random admittances
	// and initializing junctions
	for (int i = 0; i < rows; ++i){
		rand_number = (rand()%20) + 2;
		gi[i] = rand_number;

		junction_sums[i] = gi[i];
		junction_list[i].push_back(i);
	}


	// Marking random components for mutations.
	bool* mutations = new bool[rows];
	int mutations_found = 0;
	int mutated_comp;

	while(mutations_found != num_mutations){
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
			junction_sums[i-1] += gi[i];
		}else{
			// Find the component to associate 
			prev_node = rand()%(i-1);
			//cout << "(" << prev_node << "," << i << ")" << endl;

			junction_list[prev_node].push_back(i);
			junction_sums[prev_node] += gi[i];
		}
	}

	fill_matrix_using_junction(h_A_cusp, h_b,
								h_left, h_principal, h_right,
								rows,
								gi, junction_list, junction_sums,
								tridiag_nnz, offdiag_nnz);


}

void read_neuron_structure(char* file_name, cusp::csr_matrix<int, float, cusp::host_memory> &h_A_cusp,
								float* h_b,
								float* h_left, float* h_principal, float* h_right,
								int rows, int &tridiag_nnz, int &offdiag_nnz){

	// Juntion list with compartment numbers.
	vector< vector<int> > junction_list(rows);
	float rand_number;

	float* gi = new float[rows];
	float* junction_sums = new float[rows];

	// Generating random admittances
	// and initializing junctions
	for (int i = 0; i < rows; ++i){
		rand_number = (rand()%20) + 2;
		gi[i] = (float)rand_number;

		junction_sums[i] = gi[i];
		junction_list[i].push_back(i);
	}

	// read from file and get junction data.
	ifstream input_file(file_name);
	string line;

	int start,end;
	int parent,child;
	float temp_float;
	while(getline(input_file, line)){
		if(line[0] != '#'){
			stringstream ss(line);
			ss >> child;
			ss >> parent;
			ss >> temp_float;
			ss >> temp_float;
			ss >> temp_float;
			ss >> temp_float;
			ss >> parent;

			if(parent != -1){
				junction_list[parent-1].push_back(child-1);
				junction_sums[parent-1] += gi[child-1];
			}

			//cout << child << " " << parent << endl;
		}
	}

	input_file.close();

	fill_matrix_using_junction(h_A_cusp, h_b,
								h_left, h_principal, h_right,
								rows,
								gi, junction_list, junction_sums,
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

	bool ANALYSIS = false;
	bool MAKE_DIAG_DOMINANT = false;
	bool FROM_FILE = true;

	int rows, num_mutations;

	int tridiag_nnz = 0;
	int offdiag_nnz = 0;

	// set stopping criteria:
    int  iteration_limit    = 500;
    float  relative_tolerance = 1e-6;

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
	cusp::csr_matrix<int, float, cusp::host_memory> h_A;

	// Pointers for tri-diagonal in host and device
	float* h_left, *h_principal, *h_right;
	float* d_left, *d_principal, *d_right;

	float* h_b;
	float* d_b;

	// Allocating memory for tri diagonal
	h_left = (float*) calloc(rows, sizeof(float));
	h_principal = (float*) calloc(rows, sizeof(float));
	h_right = (float*) calloc(rows, sizeof(float));

	cudaMalloc((void**)&d_left, rows*sizeof(float));
	cudaMalloc((void**)&d_principal, rows*sizeof(float));
	cudaMalloc((void**)&d_right, rows*sizeof(float));

	// Allocating memory for rhs
	h_b = (float*) calloc(rows+1, sizeof(float));
	cudaMalloc((void**)& d_b, rows*sizeof(float));

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
    cudaMemcpy(d_left, h_left, sizeof(float)*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_principal, h_principal, sizeof(float)*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, h_right, sizeof(float)*rows, cudaMemcpyHostToDevice);

    cudaMemcpy(d_b, h_b, sizeof(float)*rows, cudaMemcpyHostToDevice);

    // Generate device csr matrix
	cusp::csr_matrix<int, float, cusp::device_memory> d_A(h_A);

	// Solving Tri-diag(A)
    tridiagTimer.Start();
    cusparseSgtsv(cusparse_handle,
    			  h_A.num_rows,
    			  1,
    			  d_left, d_principal, d_right,
    			  d_b, h_A.num_rows);
    cudaDeviceSynchronize();
    tridiagTimer.Stop();

    float* h_tr_sol = (float*) calloc(h_A.num_rows, sizeof(float));
    cudaMemcpy(h_tr_sol, d_b, sizeof(float)* h_A.num_rows, cudaMemcpyDeviceToHost);

    // Solving CG using tridiag

	// allocate storage for solution (x) and right hand side (b)
	cusp::array1d<float, cusp::host_memory> cusp_h_x(h_A.num_rows);
	cusp::array1d<float, cusp::host_memory> cusp_h_b(h_A.num_rows);

	for (int i = 0; i < h_A.num_rows; ++i)
	{
		cusp_h_x[i] = h_tr_sol[i];
		cusp_h_b[i] = h_b[i];
	}

	//cusp::print(cusp_h_x);

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<float, cusp::device_memory> cusp_d_clever_x(cusp_h_x);
    cusp::array1d<float, cusp::device_memory> cusp_d_zero_x(h_A.num_rows, 0);

    cusp::array1d<float, cusp::device_memory> cusp_d_clever_b(cusp_h_b);
    cusp::array1d<float, cusp::device_memory> cusp_d_zero_b(cusp_h_b);


    cusp::monitor<float> cleverMonitor(cusp_d_clever_b, iteration_limit, relative_tolerance, 0, !ANALYSIS);
    cusp::monitor<float> zeroMonitor(cusp_d_zero_b, iteration_limit, relative_tolerance, 0, !ANALYSIS);

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
    

    float tridiagTime = tridiagTimer.Elapsed();
    float cuspHintTime = cuspHintTimer.Elapsed();
    float cuspZeroTime = cuspZeroTimer.Elapsed();

    float clever_time = tridiagTime+cuspHintTime;
    float speedup = cuspZeroTime/clever_time;
    int clever_iterations = cleverMonitor.iteration_count();
    int bench_iterations = zeroMonitor.iteration_count();

    float offdiag_perc = ((float)offdiag_nnz*100)/h_A.num_entries;
    float tridiag_occupancy = ((float)tridiag_nnz*100)/(3*rows-2);


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
