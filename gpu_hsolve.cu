
#include "gpu_hsolve_utils.cu"


using namespace std;

int main(int argc, char *argv[])
{

	bool ANALYSIS = true;
	bool FROM_FILE = false;

	int tridiag_nnz = 0;
	int offdiag_nnz = 0;

	// set stopping criteria:
	int  iteration_limit    = 50;
	float  relative_tolerance = 1e-6;

	GpuTimer tridiagTimer;
	GpuTimer cuspZeroTimer;
	GpuTimer cuspHintTimer;

	
	// cusparse handle
	cusparseHandle_t cusparse_handle = 0;
	cusparseCreate(&cusparse_handle);

	if(argc > 1)
		FROM_FILE = atoi(argv[1]);

	/* initialize random seed: */
	srand (time(NULL));

	// Required arrays
	double simulation_time, dT;
	int time_steps = 0;

	vector<vector<int> > junction_list;

	int num_comp = 0;
	int* h_channel_counts; // 3*i+k element means ith componenent k=0->Na k=1->K k=2->cl

	double* h_V,*h_Cm, *h_Ga, *h_Rm, *h_Em;
	double* h_gate_m,*h_gate_h,*h_gate_n;
	double* h_current_inj;

	// Setting up simulation times
	simulation_time = 50;
	dT = 1;
	time_steps = simulation_time/dT;

	// Setting number of components
	if(FROM_FILE){
		num_comp = get_rows_from_file(argv[2]);
		junction_list.resize(num_comp);

		get_structure_from_neuron(argv[2], num_comp, junction_list);

	}else{
		num_comp = 5;
		junction_list.resize(num_comp);

		int num_mutations = 1;
		generate_random_neuron(num_comp, num_mutations, junction_list);
	}

	/*
	// print junction list
	for (int i = 0; i < junction_list.size(); ++i)
	{	
		cout << i << "-> ";
		for (int j = 0; j < junction_list[i].size(); ++j)
			cout << junction_list[i][j] << " ";
		cout << endl;
	}
	*/
	

	// Allocating memory
	h_V  = new double[num_comp];
	h_Cm = new double[num_comp];
	h_Ga = new double[num_comp];
	h_Rm = new double[num_comp];
	h_Em = new double[num_comp];

	h_gate_m = new double[num_comp];
	h_gate_n = new double[num_comp];
	h_gate_h = new double[num_comp];

	h_current_inj = new double[time_steps];


	//First 25% and last 25% of currents to be set
	//int temp = (25*time_steps)/100;
	for (int i = 0; i < time_steps; ++i){
		h_current_inj[i] = I_EXT;
		//h_current_inj[time_steps-i] = I_EXT;
	}


	// Setting up channels
	h_channel_counts = new int[3*num_comp];

	// Randomly assigning channel types for chann in compartment.
	for (int i = 0; i < num_comp; ++i)
	{
		int num_channels = max(3,rand()%MAX_CHAN_PER_COMP);
		// Making sure compartment has atleast one Na,K,Cl channel
		int chan_type, na_count = 1, k_count = 1, cl_count = 1;
		for (int j = 0; j < num_channels-3; ++j)
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

		h_channel_counts[i*3] = na_count;
		h_channel_counts[i*3+1] = k_count;
		h_channel_counts[i*3+2] = cl_count;
	}


	populate_V(h_V, num_comp);
	populate_Cm(h_Cm, num_comp);
	populate_Ga(h_Ga, num_comp);
	populate_Rm(h_Rm, num_comp);
	populate_Em(h_Em, num_comp);

	// ****************************** SetUp Matrix ************************************
	
	// Passive data cm/dt + 1/Rm + G
	cusp::csr_matrix<int, double, cusp::host_memory> h_A_cusp;
	double* h_b;
	double* h_maindiag_passive;
	double* h_tridiag_data;
	int* h_maindiag_map;

	// Allocating memory
	h_b = new double[num_comp];
	h_maindiag_passive = new double[num_comp];
	h_tridiag_data = new double[3*num_comp];
	h_maindiag_map = new int[num_comp];

	fill_matrix_using_junction(num_comp, junction_list,
								h_A_cusp, h_b,
								h_maindiag_passive, h_tridiag_data, h_maindiag_map,
								h_Cm, h_Ga, h_Rm, dT,
								tridiag_nnz, offdiag_nnz);
	// **************************** Device memory Allocation ****************************
	double* d_V,*d_Cm, *d_Rm, *d_Em;
	double* d_gate_m,*d_gate_h,*d_gate_n;
	double* d_current_inj;
	//double* d_b;
	cusp::array1d<double,cusp::device_memory> d_b_cusp(num_comp);

	int* d_channel_counts;

	cusp::csr_matrix<int, double, cusp::device_memory> d_A_cusp(h_A_cusp);
	double* d_maindiag_passive, *d_tridiag_data;
	int* d_maindiag_map;

	cudaMalloc((void**)&d_V, num_comp*sizeof(double));
	cudaMalloc((void**)&d_Cm, num_comp*sizeof(double));
	cudaMalloc((void**)&d_Rm, num_comp*sizeof(double));
	cudaMalloc((void**)&d_Em, num_comp*sizeof(double));

	cudaMalloc((void**)&d_gate_m, num_comp*sizeof(double));
	cudaMalloc((void**)&d_gate_h, num_comp*sizeof(double));
	cudaMalloc((void**)&d_gate_n, num_comp*sizeof(double));

	cudaMalloc((void**)&d_current_inj, time_steps*sizeof(double));
	//cudaMalloc((void**)&d_b, num_comp*sizeof(double));

	cudaMalloc((void**)&d_channel_counts, (3*num_comp)*sizeof(int));

	cudaMalloc((void**)&d_maindiag_passive, num_comp*sizeof(double));
	cudaMalloc((void**)&d_tridiag_data, (3*num_comp)*sizeof(double));
	cudaMalloc((void**)&d_maindiag_map, num_comp*sizeof(int));

	cudaMemcpy(d_V, h_V, sizeof(double)*num_comp, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Cm, h_Cm, sizeof(double)*num_comp, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Rm, h_Rm, sizeof(double)*num_comp, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Em, h_Em, sizeof(double)*num_comp, cudaMemcpyHostToDevice);

	cudaMemcpy(d_gate_m, h_gate_m, sizeof(double)*num_comp, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gate_h, h_gate_h, sizeof(double)*num_comp, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gate_n, h_gate_n, sizeof(double)*num_comp, cudaMemcpyHostToDevice);

	cudaMemcpy(d_current_inj, h_current_inj, sizeof(double)*time_steps, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_b, h_b, sizeof(double)*num_comp, cudaMemcpyHostToDevice);

	//for (int i = 0; i < num_comp; ++i) cout << h_channel_counts[3*i] << " " << h_channel_counts[3*i+1] << " " << h_channel_counts[3*i+2] << endl;
	cudaMemcpy(d_channel_counts, h_channel_counts, sizeof(int)*(3*num_comp), cudaMemcpyHostToDevice);

	cudaMemcpy(d_maindiag_passive, h_maindiag_passive, sizeof(double)*num_comp, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tridiag_data, h_tridiag_data, sizeof(double)*(3*num_comp), cudaMemcpyHostToDevice);
	cudaMemcpy(d_maindiag_map, h_maindiag_map, sizeof(int)*num_comp, cudaMemcpyHostToDevice);

	// Extra memory for performing simulation.
	double* d_GkSum, *d_GkEkSum;
	cudaMalloc((void**)&d_GkSum, num_comp*sizeof(double));
	cudaMemset(d_GkSum,0, sizeof(double) * num_comp);
	cudaMalloc((void**)&d_GkEkSum, num_comp*sizeof(double));
	cudaMemset(d_GkEkSum,0, sizeof(double) * num_comp);


	// **************************** Simulation begins ************************************
	double h_Vplot[time_steps*num_comp];

	for (int i = 0; i < time_steps; ++i)
	{
		// Each run of simulation

		// ADVANCE m,n,h channels
		advance_channel_m<<<1,num_comp>>>(d_V, d_gate_m, dT);
		advance_channel_n<<<1,num_comp>>>(d_V, d_gate_h, dT);
		advance_channel_h<<<1,num_comp>>>(d_V, d_gate_n, dT);
		cudaDeviceSynchronize();
	
		// CALCULATE Gk and GkEk values
		calculate_gk_gkek_sum<<<1,num_comp>>>(d_V, 
											d_gate_m, d_gate_h, d_gate_n, 
											d_channel_counts, 
											d_GkSum, d_GkEkSum);
		cudaDeviceSynchronize();

		// CALCULATE currents
		double externalCurrent = h_current_inj[i];
		calculate_currents<<<1,num_comp>>>(d_V, d_Cm, dT, 
							d_Em, d_Rm, 
							d_GkEkSum, externalCurrent, 
							thrust::raw_pointer_cast(&d_b_cusp[0]));
		cudaDeviceSynchronize();

		// UPDATE matrix and TRIDIAG
		update_matrix<<<1,num_comp>>>(d_maindiag_passive, d_GkSum, 
					 d_maindiag_map, 
					 thrust::raw_pointer_cast(&(d_A_cusp.values[0])), d_tridiag_data);

		//cudaMemcpy(h_b, d_b, num_comp* sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		// Cloning b,because tri diagonal solver overrites it with answer
		cusp::array1d<double,cusp::device_memory> d_b_cusp_copy1(d_b_cusp);
		cusp::array1d<double,cusp::device_memory> d_b_cusp_copy2(d_b_cusp);
		cusp::array1d<double, cusp::device_memory> d_x_zero_cusp(num_comp, 0);

		//cusp::print(d_A_cusp);
		//cusp::print(d_b_cusp);
		// Solver
		tridiagTimer.Start();
		cusparseDgtsv(cusparse_handle,
					  num_comp,
					  1,
					  d_tridiag_data, &d_tridiag_data[num_comp], &d_tridiag_data[num_comp*2],
					  thrust::raw_pointer_cast(&d_b_cusp[0]), num_comp);
		cudaDeviceSynchronize();
		tridiagTimer.Stop();

		//cusp::print(d_b_cusp);

		cusp::monitor<double> cleverMonitor(d_b_cusp_copy1, iteration_limit, relative_tolerance, 0, !ANALYSIS);
		cusp::monitor<float> zeroMonitor(d_b_cusp_copy2, iteration_limit, relative_tolerance, 0, !ANALYSIS);

		// solve the linear system A * x = b with the Conjugate Gradient method
		cuspZeroTimer.Start();
			cusp::krylov::gmres(d_A_cusp, d_x_zero_cusp, d_b_cusp_copy1, iteration_limit, zeroMonitor);
			//cusp::krylov::cg(d_A_cusp, d_x_zero_cusp, d_b_cusp_copy1, zeroMonitor);
			cudaDeviceSynchronize();
		cuspZeroTimer.Stop();

		// solve the linear system A * x = b with the Conjugate Gradient method
		cuspHintTimer.Start();
			cusp::krylov::gmres(d_A_cusp, d_b_cusp, d_b_cusp_copy2, iteration_limit, cleverMonitor);
			//cusp::krylov::cg(d_A_cusp, d_b_cusp, d_b_cusp_copy2, zeroMonitor);
			cudaDeviceSynchronize();
		cuspHintTimer.Stop();

		// UPDATE V
		update_V<<<1,num_comp>>>(thrust::raw_pointer_cast(&d_b_cusp_copy2[0]), d_V);
		cudaDeviceSynchronize();

		// Transfer V to cpu for plotting
		cudaMemcpy(&h_Vplot[num_comp*i], d_V, num_comp* sizeof(double), cudaMemcpyDeviceToHost);
		
		// Timings
		float tridiagTime = tridiagTimer.Elapsed();	
		float cuspHintTime = cuspHintTimer.Elapsed();
		float cuspZeroTime = cuspZeroTimer.Elapsed();

		float clever_time = tridiagTime+cuspHintTime;
		float speedup = cuspZeroTime/clever_time;
		int clever_iterations = cleverMonitor.iteration_count();
		int bench_iterations = zeroMonitor.iteration_count();


		printf("Speedup %.2f\n",speedup);
		printf("Clever Time %.2f %d (%.2f + %.2f)\n", clever_time, clever_iterations, tridiagTime, cuspHintTime);
		printf("Bench  Time %.2f %d \n", cuspZeroTime, bench_iterations);

	}


	// Printing v value of first compartment.
	for (int i = 0; i < time_steps; ++i)
		cout << i << "," << h_Vplot[num_comp*i] << endl;
	
	

	return 0;
}