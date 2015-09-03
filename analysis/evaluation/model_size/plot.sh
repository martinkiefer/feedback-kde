#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../conf.sh

$PYTHON $DIR/extract.py --file $DIR/result.csv

gnuplot $DIR/plot.gnuplot > $DIR/model.pdf
