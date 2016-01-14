set datafile separator ","
set terminal png

plot for [i=2:40] 'file_voltage.csv' using 1:i with lines notitle


set output 'allv.png'
replot
