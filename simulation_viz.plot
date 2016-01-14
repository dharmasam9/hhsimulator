set datafile separator ","
set terminal png

set y2tics

plot 'file_voltage.csv' using 1:2 title 'voltage' with lines,\
	 'file_main_diag.csv' using 1:2 title 'MainDiagonal'  with lines,\
	 'file_b_vector.csv' using 1:2 title 'B vector' with lines axes x1y2
set output 'simulation_viz_voltage.png'
replot


plot 'file_solver.csv' using 1:9 title 'Iteration saving' with lines,\
	 'file_solver.csv' using 1:2 title 'Speedup' with lines axes x1y2
set output 'simulation_viz_savediters.png'
replot


#plot 'output.csv' using 4:xtic(1) with lines
#set output 'simulation_viz_bvecvalue.png'
#replot

#plot 'output.csv' using 11:xtic(1) with lines
#set output 'simulation_viz_speedup.png'
#replot
