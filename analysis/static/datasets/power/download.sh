#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../../conf.sh

# Now check whether the file already exists.
if [ ! -f $DIR/raw/data.csv ] ; then
  mkdir -p $DIR/raw
  cd /tmp
  wget -O /tmp/household_power_consumption.zip http://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
  unzip -o /tmp/household_power_consumption.zip
  cd -
  # Remove the CSV header.
  tail -n +2 /tmp/household_power_consumption.txt > /tmp/data.csv
  # Transform date and time information.
  $PYTHON $DIR/convert.py /tmp/data.csv > $DIR/raw/data.csv
  # And change the seperation character to |.
  sed -i -e 's/;/|/g' $DIR/raw/data.csv
fi
