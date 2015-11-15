#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>

#include <math.h>

using namespace std;

float simulationTime = 100; // Time of simulation in milli second
float deltaT = .01; // Time step in milli seconds

float gbar_K = 36;
float gbar_Na = 120;
float gbar_L = 0.3;

float E_Na = 115;
float E_K = -12;
float E_L = 10.6;
float C_m = 1;


inline float alpha_m_ofV(float V){
	return .1*( (25-V) / (exp((25-V)/10)-1) );
}

inline float beta_m_ofV(float V){
	return 4*exp(-V/18);
}

inline float alpha_h_ofV(float V){
	return .07*exp(-V/20);
}

inline float beta_h_ofV(float V){
	return 1/(exp((30-V)/10)+1);
}

inline float alpha_n_ofV(float V){
	return .01 * ( (10-V) / (exp((10-V)/10)-1) );
}

inline float beta_n_ofV(float V){
	return .125*exp(-V/80);
}


int main(int argc, char const *argv[])
{

	ofstream V_file;
	V_file.open("data.txt");


	float I_Ext = 50;

	// Setting initial states
	float V = 0;

	float alpha_m, beta_m;
	float alpha_h, beta_h;
	float alpha_n, beta_n;

	float I_Na, I_K, I_L, I_Ion;

	float m = alpha_m/(alpha_m+beta_m);
	float h = alpha_h/(alpha_h+beta_h);
	float n = alpha_n/(alpha_n+beta_n);

	int num_steps = simulationTime/deltaT;
	float* currents = new float[num_steps];

	// First first 500 steps let current be 50
	fill(currents, currents+500, I_Ext);
	fill(currents+1500, currents+num_steps-1,50);

	for (int i = 0; i < num_steps; ++i)
	{
		V_file << i*deltaT << "," << V << endl;

		alpha_m = alpha_m_ofV(V);
		beta_m = beta_m_ofV(V);
		alpha_h = alpha_h_ofV(V);
		beta_h = beta_h_ofV(V);
		alpha_n = alpha_n_ofV(V);
		beta_n = beta_n_ofV(V);	

		I_Na = gbar_Na * pow(m,3) * h * (V-E_Na);
		I_K =  gbar_K * pow(n,4) * (V-E_K);
		I_L = gbar_L * (V-E_L);

		I_Ion = currents[i] - I_Na - I_K - I_L;

		V += deltaT*I_Ion/C_m;
		
		m +=  deltaT*(alpha_m*(1-m) - beta_m*m);
		h +=  deltaT*(alpha_h*(1-h) - beta_h*h);
		n +=  deltaT*(alpha_n*(1-n) - beta_n*n);

	}



	return 0;
}
