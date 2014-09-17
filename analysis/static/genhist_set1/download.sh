#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Prepare the tables.
mkdir -p $DIR/raw
python $DIR/generator1.py $DIR/raw
