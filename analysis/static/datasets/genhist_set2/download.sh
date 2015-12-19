#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../../conf.sh

# Prepare the tables.
mkdir -p $DIR/raw
echo "Generating data for Genhist_2, 2D"
$PYTHON $DIR/generator2.py --dim 2 --outfile $DIR/raw/gen2_d2.csv 
echo "Generating data for Genhist_2, 3D"
$PYTHON $DIR/generator2.py --dim 3 --outfile $DIR/raw/gen2_d3.csv 
echo "Generating data for Genhist_2, 5D"
$PYTHON $DIR/generator2.py --dim 5 --outfile $DIR/raw/gen2_d5.csv 
echo "Generating data for Genhist_2, 8D"
$PYTHON $DIR/generator2.py --dim 8 --outfile $DIR/raw/gen2_d8.csv 
