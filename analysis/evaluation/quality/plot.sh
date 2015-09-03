#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../conf.sh

$PYTHON $DIR/extract.py --file $DIR/result.csv

gnuplot $DIR/8.gnuplot > $DIR/8.pdf
gnuplot $DIR/3.gnuplot > $DIR/3.pdf
