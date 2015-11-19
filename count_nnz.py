#!/usr/bin/python

import sys,getopt

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1



filename = str(sys.argv[1])
print filename
line_count = file_len(filename)
comp_count = 0
max_value = 0

junctions = [0]*line_count

f = open(filename)
for line in f:
	if line[0] != '#':
		comp_count += 1
		values = line.split()
		last_value = int(values[len(values)-1])
		if last_value != -1:
			junctions[last_value] += 1
			if junctions[last_value] > max_value:
				max_value = junctions[last_value]
	

# resizing compartments
junctions = junctions[:comp_count+1]

# getting compartment count
off_elements = 0;
branching_count = 0;
for x in junctions:
	if x > 1:
		off_elements += (x*(x+1))/2-1
		branching_count += (x-1)

print "components: " , comp_count
print "offelemnts: " , off_elements
print "mutation %: " , branching_count
