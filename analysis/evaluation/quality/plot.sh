#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../conf.sh

cd $DIR
$PYTHON extract.py --file result.csv

gnuplot 3.gnuplot > 3.pdf
gnuplot 8.gnuplot > 8.pdf
