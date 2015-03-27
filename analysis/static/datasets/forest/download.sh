#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Now check whether the file already exists.
if [ ! -f $DIR/raw/data.csv ] ; then
  mkdir -p $DIR/raw
  cd /tmp
  wget -O /tmp/covtype.data.gz https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
  gunzip -f /tmp/covtype.data.gz
  cd -
  # Extract the first 10 columns.
  cat /tmp/covtype.data | cut -d , -f 1-10 > $DIR/raw/data.csv
  # And change the seperation character into the pipe.
  sed -i -e 's/,/|/g' $DIR/raw/data.csv
fi
