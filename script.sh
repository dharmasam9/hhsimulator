#!/bin/bash
dirr="./neurons/"

# configuration 
ROWS=1000
MUTATION_PERC=1
ITERATION_LIMIT=500
EPSILON=3
BAND_OCCUPANCY=('0')

# getting file values if f
files=""
matrix_count=0
sample_count=${#BAND_OCCUPANCY[@]}

if [ -d $dirr ]; then
    for f in "$dirr"/*; 
    	do
	    	if [ -f $f ]; then
		        files="$files $f"
		        matrix_count=$(($matrix_count + 1))
			fi
		done
else
    echo $dirr " is Not a directory"
fi


matrix_count=10

matrices=($files)
# running the alog with different dia fill percentages
for ((  i = 0 ;  i < matrix_count;  i++  ))
do
	for ((  j = 0 ;  j < sample_count;  j++  ))
		do
		#./quasi "1" "${matrices[i]}" "${BAND_OCCUPANCY[j]}" "$ONLY_MAIN_BAND" "$DEVICE" "$FOR_ANALYSIS"  "$USE_OVERLAP" "$VERIFY" "$DEBUG">> "$result"
		#./neuron "1" "${matrices[i]}" "1" "$ITERATION_LIMIT" "$EPSILON"
		./neuron "0" "$ROWS" "$MUTATION_PERC" "1" "$ITERATION_LIMIT" "$EPSILON"
		ROWS=$(($ROWS + 2000))	    
		done
	#echo $'\r' >> "$result"
done
