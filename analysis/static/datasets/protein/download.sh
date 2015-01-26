#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Now check whether the file already exists.
if [ ! -f $DIR/raw/data.csv ] ; then
	mkdir -p $DIR/raw
   wget -O /tmp/CASP.csv http://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv
	# Remove the first line.
	tail -n +2 /tmp/CASP.csv > $DIR/raw/data.csv
	# And change the seperation character into the pipe.
	sed -i -e 's/,/|/g' $DIR/raw/data.csv
fi
