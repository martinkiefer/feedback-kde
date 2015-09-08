#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../conf.sh

cd $DIR
$PYTHON extract.py --file result_time.csv --queries 100
$PYTHON extract.py --file result_stholes.csv --queries 100

gnuplot 8_plot.gnuplot > model.pdf
