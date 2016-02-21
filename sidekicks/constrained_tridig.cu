/*
	Generates a random neuron structure and tests optimizations.
*/

#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <vector>
#include <iostream>
#include <algorithm>

#include <cstdlib>
#include <time.h>

#include <cusp/krylov/gmres.h>
#include <cusp/krylov/cg.h>
#include <cusp/monitor.h>
#include <cusp/print.h>

#include "gpu_timer.h"
using namespace std;

void generate_constrained_tridiag_matrix(float* h_values, int* h_colIndex, int* h_rowPtr,
										 float* h_b,
										 float* h_left, float* h_principal, float* h_right,
										 int rows, int num_mutations, bool MAKE_DIAG_DOMINANT){



	// generate num_mutations random number in range (0,rows-1)

	vector<pair<int,int> > mutations;
	vector<bool> is_mutation(rows, false);
	bool is_done = false;

	int new_row, new_col;
	int elem_location, trans_elem_location;
	int rand_value;

	// Generating mutations.
	while(mutations.size() != 2*num_mutations){
		// generate a number between 1:rows-2
		int row =  (rand()%(rows-2))+2;
		if(!is_mutation[row]){
			is_mutation[row] = true;

			new_row = rand()%(row-1);
			new_col = row;

			elem_location = new_row*rows + new_col;
			trans_elem_location = new_col*rows + new_row;

			rand_value  = rand()%10 + 2;

			mutations.push_back(make_pair(elem_location, rand_value));
			mutations.push_back(make_pair(trans_elem_location, rand_value));
		}
	}

	// Generating non mutated part of tri-digaonal matrix
	for (int i = 1; i < rows; ++i)
	{
		if(!is_mutation[i]){
			rand_value  = rand()%10 + 2;
			mutations.push_back(make_pair((i-1)*rows + i, rand_value));
			mutations.push_back(make_pair(i*rows + (i-1), rand_value));
		}
	}

	// Pushing the principal diagonal
	for (int i = 0; i < rows; ++i)
	{	
		rand_value  = rand()%10 + 2;
		mutations.push_back(make_pair(i*rows+i, rand_value));	
	}
	
	// Sort these mutations
	sort(mutations.begin(), mutations.end());

	// cout << (3*rows-2) << " " << mutations.size() << endl;

	vector<int> row_sums(rows,0);
	vector<pair<int,int> > main_indices;

	int r,c, value;
	for (int i = 0; i < mutations.size(); ++i)
	{
		r = (mutations[i].first)/rows;
		c = (mutations[i].first)%rows;

		h_colIndex[i] = c;
		h_rowPtr[r]++;
		h_values[i] = mutations[i].second;

		row_sums[r] += mutations[i].second;

		if(r==c){
			main_indices.push_back(make_pair(r,i));
		}else{
			// Filling left and right diagonals
			if(r==c+1)
				h_left[r] = mutations[i].second;
			if(c==r+1)
				h_right[r] = mutations[i].second;
		}
	}

	int temp;
	int sum = 0;
	// Scan operation on rowPtr;
	for (int i = 0; i < rows+1; ++i)
	{
		temp = h_rowPtr[i];
		h_rowPtr[i] = sum;
		sum += temp;
	}

	if(MAKE_DIAG_DOMINANT){
		// Making it diagonally dominant.
		for (int i = 0; i < main_indices.size(); ++i)
		{
			h_values[main_indices[i].second] = row_sums[main_indices[i].first];
			// Filling main diagonal
			h_principal[main_indices[i].first] = row_sums[main_indices[i].first];
		}
	}else{
		// Making it diagonally dominant.
		for (int i = 0; i < main_indices.size(); ++i)
		{
			rand_value  = rand()%10 + 2;
			h_values[main_indices[i].second] = rand_value;
			// Filling main diagonal
			h_principal[main_indices[i].first] = rand_value;
		}

	}
	
	// cout << rows << " " << main_indices.size() << endl;

	// Setting b values.
	for (int i = 0; i < rows; ++i)
	{
		h_b[i] = 1;
	}
	


}

void print_matrix(float* values, int* colIndex, int* rowPtr, int rows){
	for (int i = 0; i < rows; ++i)
	{
		int j = 0;
		int k = rowPtr[i];

		while(j < rows){
			if(colIndex[k] == j){
				if(values[k] < 10)
					cout << "0" << values[k] << " ";
				else
					cout << values[k] << " ";
				k++;
			}else{
				cout << "00" << " ";
			}

			j++;
		}
		cout << endl;
	}
	cout << endl;
}

void convert_csr_to_cusp(float* values, int* colIndex, int* rowPtr, int rows, int nnz,
						 cusp::csr_matrix<int, float, cusp::host_memory> &h_cusp_mat){

	h_cusp_mat.resize(rows, rows, nnz);

    for (int i = 0; i < h_cusp_mat.num_entries; ++i)
    {
        h_cusp_mat.column_indices[i] = colIndex[i];
        h_cusp_mat.values[i] = values[i];
    }

    for (int i = 0; i <= h_cusp_mat.num_rows; ++i)
    {
        h_cusp_mat.row_offsets[i] = rowPtr[i];
    }
}


int main(int argc, char const *argv[])
{

	bool ANALYSIS = true;
	bool MAKE_DIAG_DOMINANT = false;

	GpuTimer tridiagTimer;
	GpuTimer cuspZeroTimer;
	GpuTimer cuspHintTimer;

    // cusparse handle
    cusparseHandle_t cusparse_handle = 0;
    cusparseCreate(&cusparse_handle);


	srand(time(NULL));
	// Create a synthetic constrained quasi tri-diagonal matrix.

	// Pointers for tri-diagonal in host and device
	float* h_left, *h_principal, *h_right;
	float* d_left, *d_principal, *d_right;

	// Pointers for quasi part in host and device.
	int nnz;
	float* h_values;
	int* h_colIndex,*h_rowPtr;
	float* h_b;	

	float* d_values;
	int* d_colIndex, *d_rowPtr;
	float* d_b;

	int rows = atoi(argv[1]);
	int num_mutations = (atoi(argv[2])*(rows-2))/100;

	//cout << num_mutations << endl;

	// Allocating memory for tri diagonal
	h_left = (float*) calloc(rows, sizeof(float));
	h_principal = (float*) calloc(rows, sizeof(float));
	h_right = (float*) calloc(rows, sizeof(float));

	cudaMalloc((void**)&d_left, rows*sizeof(float));
	cudaMalloc((void**)&d_principal, rows*sizeof(float));
	cudaMalloc((void**)&d_right, rows*sizeof(float));

	// Allocating memory for quasi part
	nnz = (3*rows-2);
	h_values = (float*) calloc(nnz, sizeof(float));
	h_colIndex = (int*) calloc(nnz, sizeof(int));
	h_rowPtr = (int*) calloc(rows+1, sizeof(int));
	h_b = (float*) calloc(rows+1, sizeof(float));

	cudaMalloc((void**)& d_values, rows*sizeof(float));
	cudaMalloc((void**)& d_colIndex, rows*sizeof(int));
	cudaMalloc((void**)& d_rowPtr, rows*sizeof(int));
	cudaMalloc((void**)& d_b, rows*sizeof(float));

	generate_constrained_tridiag_matrix(h_values, h_colIndex, h_rowPtr, h_b,  h_left, 
		h_principal, h_right, rows, num_mutations, MAKE_DIAG_DOMINANT);

	/*
	print_matrix(h_values, h_colIndex, h_rowPtr, rows);

	for (int i = 0; i < rows; ++i)
	{
		cout << h_b[i] << endl;
	}

	for (int i = 0; i < rows; ++i)
	{
		
		cout << h_left[i] << " ";
		cout << h_principal[i] << " ";
		cout << h_right[i] << " " << endl;

	}
	*/

	// Copy to gpu
    cudaMemcpy(d_left, h_left, sizeof(float)*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_principal, h_principal, sizeof(float)*rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, h_right, sizeof(float)*rows, cudaMemcpyHostToDevice);

    cudaMemcpy(d_b, h_b, sizeof(float)*rows, cudaMemcpyHostToDevice);

	cusp::csr_matrix<int, float, cusp::host_memory> h_A;
	convert_csr_to_cusp(h_values, h_colIndex, h_rowPtr, rows, nnz, h_A);
	cusp::csr_matrix<int, float, cusp::device_memory> d_A(h_A);


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

    // set stopping criteria:
    int  iteration_limit    = 500;
    float  relative_tolerance = 1e-6;
    int restart = 500;

    cusp::monitor<float> cleverMonitor(cusp_d_clever_b, iteration_limit, relative_tolerance, 0, !ANALYSIS);
    cusp::monitor<float> zeroMonitor(cusp_d_zero_b, iteration_limit, relative_tolerance, 0, !ANALYSIS);

    // solve the linear system A * x = b with the Conjugate Gradient method
    cuspZeroTimer.Start();
        //cusp::krylov::gmres(d_A, cusp_d_zero_x, cusp_d_zero_b, restart, zeroMonitor);
        cusp::krylov::cg(d_A, cusp_d_zero_x, cusp_d_zero_b, zeroMonitor);
        cudaDeviceSynchronize();
    cuspZeroTimer.Stop();

    cuspHintTimer.Start();
    	//cusp::krylov::gmres(d_A, cusp_d_clever_x, cusp_d_clever_b, restart, cleverMonitor);
        cusp::krylov::cg(d_A, cusp_d_clever_x, cusp_d_clever_b, cleverMonitor);
    	cudaDeviceSynchronize();
    cuspHintTimer.Stop();


    float tridiagTime = tridiagTimer.Elapsed();
    float cuspHintTime = cuspHintTimer.Elapsed();
    float cuspZeroTime = cuspZeroTimer.Elapsed();

    float clever_time = tridiagTime+cuspHintTime;
    float speedup = cuspZeroTime/clever_time;
    int clever_iterations = cleverMonitor.iteration_count();
    int bench_iterations = zeroMonitor.iteration_count();


    if(ANALYSIS){
        cout << speedup << " " << clever_time << " " << cuspZeroTime << " " << clever_iterations << " " << bench_iterations << " " << tridiagTime << " " << cuspHintTime << endl;
    }else{
        cout <<  speedup << endl;
        cout << "My time: " << tridiagTime+cuspHintTime << " (" << tridiagTime << "," << cuspHintTime << ")" << " in iterations " << cleverMonitor.iteration_count() <<  endl;
        cout << "Bnc tym: " << cuspZeroTime << " in iterations " << zeroMonitor.iteration_count() << endl;    
    }




	return 0;
}

