#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Now check whether the file already exists.
if [ ! -f $DIR/raw/data.csv ] ; then
   mkdir -p $DIR/raw
   wget -O /tmp/bike.zip http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip
   unzip -u /tmp/bike.zip -d /tmp
   # Remove the CSV header.
   tail -n +2 /tmp/hour.csv > /tmp/data.csv
   # Transform dates into integers.
   python $DIR/convert.py /tmp/data.csv > $DIR/raw/data.csv
   # And change the seperation character to pipe.
   sed -i -e 's/,/|/g' $DIR/raw/data.csv
fi
