
#include "gpu_hsolve_utils.cu"


using namespace std;

int main(int argc, char *argv[])
{

	bool ANALYSIS = true;
	bool DEBUG = false;
	bool FROM_FILE = false;

	int tridiag_nnz = 0;
	int offdiag_nnz = 0;

	// set stopping criteria:
	int  iteration_limit    = 50;
	float  relative_tolerance = 1e-6;

	// cusparse handle
	cusparseHandle_t cusparse_handle = 0;
	cusparseCreate(&cusparse_handle);

	/* initialize random seed: */
	srand (time(NULL));

	// Required arrays
	double simulation_time = 0.03, dT = 0.01;
	int time_steps = 0;

	vector<vector<int> > junction_list;

	int num_comp = 0;
	int* h_channel_counts; // 3*i+k element means ith componenent k=0->Na k=1->K k=2->cl

	double* h_V,*h_Cm, *h_Ga, *h_Rm, *h_Em;
	double* h_gate_m,*h_gate_h,*h_gate_n;
	double* h_current_inj;
	int skip_stride = 1;

	// If from file read the structure or generate structure
	if(argc > 1)
		FROM_FILE = atoi(argv[1]);

	// Setting number of components
	if(FROM_FILE){
		num_comp = get_rows_from_file(argv[2]);

		if(argc > 3)
			simulation_time = atof(argv[3]);

		if(argc > 4)
			dT = atof(argv[4]);

		if(argc > 5)
			skip_stride = atoi(argv[5]);

		if(argc > 6)
			DEBUG = atoi(argv[6]);

		if(argc > 7)
			iteration_limit = atoi(argv[7]);

		if(argc > 8)
			relative_tolerance = pow(10,-1*atoi(argv[8]));


		junction_list.resize(num_comp);
		get_structure_from_neuron(argv[2], num_comp, junction_list);

	}else{
		int num_mutations = 1;
		num_comp = 5;

		if(argc > 2)
			num_comp = atoi(argv[2]);		

		if(argc > 3)
			num_mutations = (atof(argv[3])*(num_comp-2))/100; // Branching percentage

		if(argc > 4)
			simulation_time = atof(argv[4]);

		if(argc > 5)
			dT = atof(argv[5]);

		if(argc > 6)
			skip_stride = atoi(argv[6]);

		if(argc > 7)
			DEBUG = atoi(argv[7]);

		if(argc > 8)
			iteration_limit = atoi(argv[8]);

		if(argc > 9)
			relative_tolerance = pow(10,-1*atoi(argv[9]));

		junction_list.resize(num_comp);
		generate_random_neuron(num_comp, num_mutations, junction_list);
	}

	// Calculating time_steps
	time_steps = simulation_time/dT;

	// Allocating memory
	h_V  = new double[num_comp]();
	h_Cm = new double[num_comp]();
	h_Ga = new double[num_comp]();
	h_Rm = new double[num_comp]();
	h_Em = new double[num_comp]();

	h_gate_m = new double[num_comp]();
	h_gate_n = new double[num_comp]();
	h_gate_h = new double[num_comp]();

	h_current_inj = new double[time_steps]();
	h_channel_counts = new int[3*num_comp]();


	// Initializing m,h,h
	initialize_gates(num_comp, h_gate_m, h_gate_n, h_gate_h);

	// Full current through out.
	for (int i = 0; i < time_steps; ++i){
		h_current_inj[i] = I_EXT;
		//h_current_inj[i] = 0;		
	}



	/* Managing current
	int temp = (25*time_steps)/100;
	for (int i = 0; i < temp; ++i){
		h_current_inj[i] = I_EXT;
		h_current_inj[time_steps-temp-i] = I_EXT;
	}
	*/

	// Setting up channels
	// Randomly assigning channel types for chann in compartment.
	for (int i = 0; i < num_comp; ++i)
	{
		int num_channels;
		int chan_type, na_count, k_count, cl_count;

		bool fill_random = false;
		
		if(fill_random){
			num_channels = max(3,rand()%MAX_CHAN_PER_COMP);
			// Making sure compartment has atleast one Na,K,Cl channel
			na_count = 1; k_count = 1; cl_count = 1;
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
		}else{
			// 60-30-10 proportions of Na,K,Cl
			na_count = 3;
			k_count = 3;
			cl_count = 3;
		}
		
		/*
		if(i == 0)
			cout << na_count << " " << k_count << " " << cl_count << endl;
		*/

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
	h_b = new double[num_comp]();
	h_maindiag_passive = new double[num_comp]();
	h_tridiag_data = new double[3*num_comp]();
	h_maindiag_map = new int[num_comp]();

	fill_matrix_using_junction(num_comp, junction_list,
								h_A_cusp, h_b,
								h_maindiag_passive, h_tridiag_data, h_maindiag_map,
								h_Cm, h_Ga, h_Rm, dT,
								tridiag_nnz, offdiag_nnz);
	// **************************** Device memory Allocation ****************************
	double* d_V,*d_Cm, *d_Rm, *d_Em;
	double* d_gate_m,*d_gate_h,*d_gate_n;
	double* d_current_inj;
	int* d_channel_counts;
	cusp::array1d<double,cusp::device_memory> d_tridiag_sol(num_comp,0);
	cusp::array1d<double,cusp::device_memory> d_b_cusp(num_comp,0);

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

	//print_matrix(h_A_cusp);	
	
	// STATE BEFORE SIMULATION
	/*
	if(DEBUG){
		cusp::print(d_A_cusp);
		cudaMemcpy(h_tridiag_data, d_tridiag_data, sizeof(double)*(3*num_comp), cudaMemcpyDeviceToHost);

		for (int i = 0; i < num_comp; ++i)
			cout << h_tridiag_data[i] << " " << h_tridiag_data[num_comp+i] << " " << h_tridiag_data[2*num_comp+i] << endl;

		// Printing currents
		cusp::print(d_b_cusp);

		// Each run of simulation
		cudaMemcpy(h_V, d_V, sizeof(double)*num_comp, cudaMemcpyDeviceToHost);

		for (int i = 0; i < num_comp; ++i)
			cout << h_V[i] << endl;		

		cout << "********************************************" << endl;

	}
	*/

	// ************************************************
	if(!ANALYSIS) cout << "SIMULATION BEGINS" << endl;

	ofstream V_file, Maindiag_file, B_file;
	ofstream solver_file;

	V_file.open("file_voltage.csv");
	Maindiag_file.open("file_main_diag.csv");
	B_file.open("file_b_vector.csv");
	solver_file.open("file_solver.csv");


	double offdiag_perc = (offdiag_nnz*100.0)/h_A_cusp.num_entries;
	double tridiag_occupancy = (tridiag_nnz * 100.0)/ (3*h_A_cusp.num_rows);
	solver_file << h_A_cusp.num_rows << " " << h_A_cusp.num_entries << " " << tridiag_nnz << " " << offdiag_nnz << " " << offdiag_perc << " " << tridiag_occupancy << endl;


	double h_Vplot[num_comp];
	double h_Mplot[num_comp];
	double h_Nplot[num_comp];
	double h_Hplot[num_comp];

	int NUM_THREAD_PER_BLOCK = 512;
	int NUM_BLOCKS = ceil((num_comp*1.0)/NUM_THREAD_PER_BLOCK);

	float total_clever_time = 0;
	float total_fast_time = 0;
	float total_zero_time = 0;
	int total_clever_savings = 0;
	int total_fast_savings = 0;
	
	for (int i = 0; i < time_steps; ++i)
	{

		// GPU Timers
		GpuTimer cuspZeroTimer;
		GpuTimer tridiagTimer, cuspHintTimer;
		GpuTimer fastXTimer, cuspFastTimer;
		
		GpuTimer channelTimer;
		GpuTimer currentTimer;


		channelTimer.Start();
			// ADVANCE m,n,h channels
			advance_channel_m<<<NUM_BLOCKS,NUM_THREAD_PER_BLOCK>>>(num_comp, d_V, d_gate_m, dT);
			advance_channel_n<<<NUM_BLOCKS,NUM_THREAD_PER_BLOCK>>>(num_comp, d_V, d_gate_n, dT);
			advance_channel_h<<<NUM_BLOCKS,NUM_THREAD_PER_BLOCK>>>(num_comp, d_V, d_gate_h, dT);
			cudaDeviceSynchronize();
		channelTimer.Stop();
		
		currentTimer.Start();
			// CALCULATE Gk and GkEk values
			calculate_gk_gkek_sum<<<NUM_BLOCKS,NUM_THREAD_PER_BLOCK>>>(num_comp, d_V, 
												d_gate_m, d_gate_h, d_gate_n, 
												d_channel_counts, 
												d_GkSum, d_GkEkSum);
			cudaDeviceSynchronize();

			// CALCULATE currents
			double externalCurrent = h_current_inj[i];
			calculate_currents<<<NUM_BLOCKS,NUM_THREAD_PER_BLOCK>>>(num_comp, d_V, d_Cm, dT, 
								d_Em, d_Rm, 
								d_GkEkSum, externalCurrent, 
								thrust::raw_pointer_cast(&d_b_cusp[0]));
			cudaDeviceSynchronize();

			// UPDATE matrix and TRIDIAG
			update_matrix<<<NUM_BLOCKS,NUM_THREAD_PER_BLOCK>>>(num_comp, num_comp, d_maindiag_passive, d_GkSum, 
						 d_maindiag_map, 
						 thrust::raw_pointer_cast(&(d_A_cusp.values[0])), d_tridiag_data);

			//cudaMemcpy(h_b, d_b, num_comp* sizeof(double), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
		currentTimer.Stop();

		// Cloning b,because tri diagonal solver overrites it with answer
		cusp::array1d<double,cusp::device_memory> d_b_cusp_copy1(d_b_cusp);
		cusp::array1d<double,cusp::device_memory> d_b_cusp_copy2(d_b_cusp);
		cusp::array1d<double,cusp::device_memory> d_b_cusp_copy3(d_b_cusp);

		cusp::array1d<double, cusp::device_memory> d_x_zero_cusp(num_comp, 0);
		cusp::array1d<double, cusp::device_memory> d_x_fast_cusp(num_comp, 0);

		cusp::array1d<double,cusp::host_memory> h_b_cusp_copy3(d_b_cusp);

		/*
		// Set to zero
		cudaMemset(thrust::raw_pointer_cast(&d_b_cusp[0]), 0, num_comp*sizeof(double));
		cudaDeviceSynchronize();
		*/

		// Solver
		
		// Calculates tri diagonal solution every skip_stride
		if(i%skip_stride == 0){
			tridiagTimer.Start();
				cusparseDgtsv(cusparse_handle,
						  num_comp,
						  1,
						  d_tridiag_data, &d_tridiag_data[num_comp], &d_tridiag_data[num_comp*2],
						  thrust::raw_pointer_cast(&d_b_cusp[0]), num_comp);
				cudaDeviceSynchronize();
			tridiagTimer.Stop();

			// Archive tridiagonal solution	
			copy_device_vector<<<NUM_BLOCKS,NUM_THREAD_PER_BLOCK>>>(num_comp, thrust::raw_pointer_cast(&d_b_cusp[0]), thrust::raw_pointer_cast(&d_tridiag_sol[0]));
			cudaDeviceSynchronize();

		}else{
			// Get tridiagonal solution from archive
			copy_device_vector<<<NUM_BLOCKS,NUM_THREAD_PER_BLOCK>>>(num_comp, thrust::raw_pointer_cast(&d_tridiag_sol[0]), thrust::raw_pointer_cast(&d_b_cusp[0]));
			cudaDeviceSynchronize();

			tridiagTimer.Start(); tridiagTimer.Stop(); // Calling for homogenity
		}
		

		cusp::monitor<double> cleverMonitor(d_b_cusp_copy1, iteration_limit, relative_tolerance, 0, !ANALYSIS);
		cusp::monitor<double> zeroMonitor(d_b_cusp_copy2, iteration_limit, relative_tolerance, 0, !ANALYSIS);
		cusp::monitor<double> fastMonitor(d_b_cusp_copy3, iteration_limit, relative_tolerance, 0, !ANALYSIS);

		// solve the linear system A * x = b with the GMRES method
		cuspZeroTimer.Start();
			cusp::krylov::gmres(d_A_cusp, d_x_zero_cusp, d_b_cusp_copy1, iteration_limit, zeroMonitor);
			//cusp::krylov::cg(d_A_cusp, d_x_zero_cusp, d_b_cusp_copy1, zeroMonitor);
			cudaDeviceSynchronize();
		cuspZeroTimer.Stop();

		// solve the linear system A * x = b with the GMRES method
		cuspHintTimer.Start();
			cusp::krylov::gmres(d_A_cusp, d_b_cusp, d_b_cusp_copy2, iteration_limit, cleverMonitor);
			//cusp::krylov::cg(d_A_cusp, d_b_cusp, d_b_cusp_copy2, zeroMonitor);
			cudaDeviceSynchronize();
		cuspHintTimer.Stop();

		// Get fast value
		fastXTimer.Start();
			compute_fast_x<<<NUM_BLOCKS, NUM_THREAD_PER_BLOCK>>>(num_comp, num_comp, d_tridiag_data, 
				thrust::raw_pointer_cast(&d_b_cusp_copy2[0]),
				thrust::raw_pointer_cast(&d_x_fast_cusp[0]));
			cudaDeviceSynchronize();
		fastXTimer.Stop();

		cuspFastTimer.Start();
			cusp::krylov::gmres(d_A_cusp, d_x_fast_cusp, d_b_cusp_copy3, iteration_limit, fastMonitor);
			cudaDeviceSynchronize();
		cuspFastTimer.Stop();

		// UPDATE V
		copy_device_vector<<<NUM_BLOCKS,NUM_THREAD_PER_BLOCK>>>(num_comp, thrust::raw_pointer_cast(&d_b_cusp[0]), d_V);
		cudaDeviceSynchronize();

		if(DEBUG && i<=10){
			print_iteration_state(d_A_cusp, d_b_cusp, d_b_cusp_copy1);
			cudaDeviceSynchronize();
		}

		// ***************************************

		// Transfer V to cpu for plotting
		cudaMemcpy(h_Vplot, d_V, num_comp* sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_Mplot, d_gate_m, num_comp* sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_Hplot, d_gate_h, num_comp* sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_Nplot, d_gate_n, num_comp* sizeof(double), cudaMemcpyDeviceToHost);

		cudaMemcpy(h_tridiag_data, d_tridiag_data, sizeof(double)*(3*num_comp), cudaMemcpyDeviceToHost);

		//cout << h_tridiag_data[i] << " " << h_tridiag_data[num_comp+i] << " " << h_tridiag_data[2*num_comp+i] << endl;
		
		// Timings
		float tridiagTime = tridiagTimer.Elapsed();	
		float cuspHintTime = cuspHintTimer.Elapsed();
		float cuspZeroTime = cuspZeroTimer.Elapsed();
		float cuspFastTime = fastXTimer.Elapsed() + cuspFastTimer.Elapsed();
		float channelTime = channelTimer.Elapsed();
		float currentTime = currentTimer.Elapsed();

		float clever_time = tridiagTime+cuspHintTime;
		float speedup = cuspZeroTime/clever_time;
		int clever_iterations = cleverMonitor.iteration_count();
		int bench_iterations = zeroMonitor.iteration_count();
		int fast_iterations = fastMonitor.iteration_count();

		float channelPerc = (channelTime * 100)/(channelTime + currentTime + clever_time);
		float currentPerc = (currentTime * 100)/(channelTime + currentTime + clever_time);
		float solverPerc = (clever_time * 100)/(channelTime + currentTime + clever_time);

		total_clever_time += clever_time;
		total_fast_time += cuspFastTime;
		total_zero_time += cuspZeroTime;
		total_clever_savings += (bench_iterations-clever_iterations);
		total_fast_savings += (bench_iterations-fast_iterations);

		if(i<0){
			printf("Speedup %.2f\n",speedup);
			printf("Clever Time %.2f %d (%.2f + %.2f)\n", clever_time, clever_iterations, tridiagTime, cuspHintTime);
			printf("Fast   Time %.2f %d (%.2f + %.2f)\n", cuspFastTime, fast_iterations, fastXTimer.Elapsed(), cuspFastTimer.Elapsed());
			printf("Bench  Time %.2f %d \n", cuspZeroTime, bench_iterations);	
			printf("profil Time %.2f %.2f %.2f\n", channelPerc, currentPerc, solverPerc);

		}

		// Capturing headers
		if(i==0){
			solver_file << "#timestep,speedup,clever_time,fast_time,zero_time,clever_iter,fast_iter,zero_iter,clever_sav,fast_sav,tridiagTime,cuspHintTime,channel%%,current%%,solver%%" << endl;
			V_file << "timestep" << ","; Maindiag_file << "timestep" << ","; B_file << "timestep" << ",";
			for(int j=0;j<num_comp;j++){
				V_file << j << ","; Maindiag_file << j << ","; B_file << j << ",";
			}
			V_file << endl; Maindiag_file << endl; B_file << endl;
		}

		// Capturing data in files.
		solver_file << i*dT << "," <<  
					speedup << "," << 
					clever_time << "," << cuspFastTime << "," << cuspZeroTime << "," << 
					clever_iterations << "," << fast_iterations << "," << bench_iterations << "," 
					<< (bench_iterations-clever_iterations) << "," << (bench_iterations-fast_iterations) << ","
					<< tridiagTime << "," << cuspHintTime << "," 
					<< channelPerc << "," << currentPerc << "," << solverPerc <<  endl;

		V_file << i*dT << ",";
		Maindiag_file << i*dT << ",";
		B_file << i*dT << ",";

		for (int j = 0; j < num_comp; ++j)
		{
			V_file << h_Vplot[j] << ",";
			Maindiag_file << h_tridiag_data[num_comp+j] << ",";
			B_file << h_b_cusp_copy3[j] << ",";
			if(j==num_comp-1){
				V_file << endl;
				Maindiag_file << endl;
				B_file << endl;
			}
		}

	}

	cout << fixed << total_clever_time << "," << total_fast_time << "," << total_zero_time << "," << total_clever_savings <<  "," << total_fast_savings <<  endl;

	V_file.close();
	Maindiag_file.close();
	B_file.close();
	solver_file.close();

	return 0;
}