reset

set datafile separator ","
set terminal png
set output 'simulation_viz_savediters.png'

set y2tics
set y2range[0:10]
unset key

plot 'file_voltage.csv' using 1:2 title 'voltage' with lines,\
	 'file_solver.csv' using 1:7 title 'Clever Itrtns' with lines axes x1y2,\
	 'file_solver.csv' using 1:9 title 'Zero Itrnts' with lines axes x1y2
	 #'file_solver.csv' using 1:8 title 'Fast Itrtns' with lines axes x1y2,\
	 #'file_solver.csv' using 1:10 title 'Clever savings' with lines axes x1y2,\
	 #'file_solver.csv' using 1:11 title 'Fast savings' with lines axes x1y2,\
	 #'file_solver.csv' using 1:2 title 'Clever Speedup' with lines axes x1y2,\
	 #'file_solver.csv' using 1:3 title 'Fast Speedup' with lines axes x1y2,\


#plot 'file_voltage.csv' using 1:2 title 'voltage' with lines,\
#	 'file_main_diag.csv' using 1:2 title 'MainDiagonal'  with lines,\
#	 'file_b_vector.csv' using 1:2 title 'B vector' with lines axes x1y2
#set output 'simulation_viz_voltage.png'
#replot


#plot 'file_solver.csv' using 1:10 title 'Clever savings' with lines,\
#	 'file_solver.csv' using 1:11 title 'Fast savings' with lines,\
#	 'file_solver.csv' using 1:2 title 'Clever Speedup' with lines axes x1y2,\
#	 'file_solver.csv' using 1:3 title 'Fast Speedup' with lines axes x1y2
#set output 'simulation_viz_savediters.png'
#replot


#plot 'output.csv' using 4:xtic(1) with lines
#set output 'simulation_viz_bvecvalue.png'
#replot

#plot 'output.csv' using 11:xtic(1) with lines
#set output 'simulation_viz_speedup.png'
#replot
