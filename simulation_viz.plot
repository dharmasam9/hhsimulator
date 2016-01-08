set datafile separator ","
set terminal png

set y2tics

plot 'output.csv' using 2:xtic(1) title 'voltage' with lines,\
	 'output.csv' using 11:xtic(1) title 'Speedup' with lines axes x1y2
set output 'simulation_viz_voltage.png'
replot

plot 'output.csv' using 3:xtic(1) title 'MainDiagonal'  with lines,\
	 'output.csv' using 4:xtic(1) title 'B vector' with lines axes x1y2
set output 'simulation_viz_maindiagvalue.png'
replot

plot 'output.csv' using 10:xtic(1) with lines
set output 'simulation_viz_savediters.png'
replot


#plot 'output.csv' using 4:xtic(1) with lines
#set output 'simulation_viz_bvecvalue.png'
#replot

#plot 'output.csv' using 11:xtic(1) with lines
#set output 'simulation_viz_speedup.png'
#replot



