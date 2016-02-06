#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../../conf.sh

# Prepare the tables.
mkdir -p $DIR/raw
echo "Generating data for Genhist_1, 3D"
$PYTHON $DIR/generator1.py --dim 3 --outfile $DIR/raw/gen1_d3.csv --cluster_width 0.055
echo "Generating data for Genhist_1, 8D"
$PYTHON $DIR/generator1.py --dim 8 --outfile $DIR/raw/gen1_d8.csv --cluster_width 0.05
