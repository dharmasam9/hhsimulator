set datafile separator ","
set terminal png

plot 'output.csv' using 2:xtic(1) with lines
set output 'simulation_viz_voltage.png'
replot

plot 'output.csv' using 9:xtic(1) with lines
set output 'simulation_viz_speedup.png'
replot

plot 'output.csv' using 8:xtic(1) with lines
set output 'simulation_viz_savediters.png'
replot