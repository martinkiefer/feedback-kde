#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../../conf.sh

# Prepare the tables.
mkdir -p $DIR/raw
$PYTHON $DIR/generator1.py $DIR/raw
