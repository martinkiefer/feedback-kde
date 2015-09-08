#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../conf.sh

cd $DIR
$PYTHON extract.py --file result.csv

gnuplot plot.gnuplot > model.pdf
