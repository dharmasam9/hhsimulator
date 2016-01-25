reset

set datafile separator ","
set terminal png
set output 'iter_behaviour.png'

set size 1,1
set multiplot
#unset key


set xlabel 'time(ms)'

# first
set origin  0,0.5
set size 1,0.5
set ylabel 'speedup'
plot 'file_solver.csv' using 1:2 title 'Clever-Speedup' with lines,\
	 'file_solver.csv' using 1:3 title 'Fast-Speedup' with lines

# second
set origin 0,0
set size 1,0.5
set y2tics
set y2range[0:10]

col_end = 100

plot for [i=2:col_end] 'file_voltage.csv' using 1:i with lines notitle,\
	 'file_solver.csv' using 1:9 title 'Zero Itrnts' with lines axes x1y2,\
	 'file_solver.csv' using 1:7 title 'Clever Itrtns' with points ps 0.5 axes x1y2,\
	 'file_solver.csv' using 1:8 title 'Fast Itrtns' with points ps 0.25 axes x1y2

unset multiplot





