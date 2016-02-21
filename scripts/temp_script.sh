./a.out "1" "$1"
gnuplot matrix_state.plot
gnuplot iter_behaviour.plot
eog matrix_state.png &
eog iter_behaviour.png


#strides=30

# running the alog with different stride values
#for ((  i = 21 ;  i <= strides;  i++  ))
#do
#	./a.out "1" "./neurons/219-1.CNG.swc" "10" "0.01" "$i"
#	#echo "1" "./neurons/219-1.CNG.swc" "10" "0.01" "$i"
#done
