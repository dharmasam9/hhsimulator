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
set y2tics
set key bottom right
set ylabel '% Iteration savings'
plot 'file_solver.csv' using 1:14 title 'Clever' with lines,\
     'file_solver.csv' using 1:15 title 'Fast' with lines

# second
set origin 0,0
set size 1,0.5
set y2tics
set key bottom horizontal right
set y2range[0:10]

col_end = 4636
stride = 20

plot for [i=2:col_end:stride] 'file_voltage.csv' using 1:i with lines notitle lc rgb 'grey',\
	 'file_solver.csv' using 1:9 title 'Zero Itrnts' with lines axes x1y2 lc rgb 'red',\
	 'file_solver.csv' using 1:8 title 'Fast Itrtns' with points ps 0.5  axes x1y2,\
	 'file_solver.csv' using 1:7 title 'Clever Itrtns' with lines lc rgb 'green' axes x1y2
	 

unset multiplot





