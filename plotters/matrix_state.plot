reset

set datafile separator ","
set terminal png
set output 'matrix_state.png'

set size 1,1
set multiplot
#unset key

col_end = 4636
stride = 20

set xlabel 'time(ms)'

# first
set size 0.5,0.5
set origin  0,0.5
set ylabel 'voltage'
plot for [i=2:col_end:stride] 'file_voltage.csv' using 1:i with lines notitle

# second
set origin 0.5,0.5
set ylabel 'Main Diag'
plot for [i=2:col_end:stride] 'file_main_diag.csv' using 1:i with lines notitle

# third
set origin 0,0
set size 1,0.5
set ylabel 'B'
plot for [i=2:col_end:stride] 'file_b_vector.csv' using 1:i with lines notitle

# fourth
#set origin 0.5,0
#set y2tics
#set y2range[0:10]
#plot 'file_voltage.csv' using 1:2 title 'voltage' with lines,\
#	 'file_solver.csv' using 1:7 title 'Clever Itrtns' with lines axes x1y2,\
#	 'file_solver.csv' using 1:9 title 'Zero Itrnts' with lines axes x1y2

unset multiplot


# set datafile separator ","
# set terminal png

# set multiplot layout 1,3 rowsfirst

# col_end = 452

# set label 1 'Vm' at graph 0.9,0.9 font ',8'
# plot for [i=2:col_end] 'file_voltage.csv' using 1:i with lines notitle

# set label 1 'main diag' at graph 0.9,0.9 font ',8'
# plot for [i=2:col_end] 'file_main_diag.csv' using 1:i with lines notitle

# set label 1 'B vector' at graph 0.9,0.9 font ',8'
# plot for [i=2:col_end] 'file_b_vector.csv' using 1:i with lines notitle


# set output 'allv.png'
# replot
