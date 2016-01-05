#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# And call the configuration script.
source $DIR/../../../conf.sh

# Check if dbgen is ready.
if [ ! -d $DIR/dbgen ] ; then
   git clone http://bitbucket.org/tkejser/tpch-dbgen.git $DIR/dbgen
   cd $DIR/dbgen
   cp ../makefile.dbgen makefile
   make
   cd -
fi

# Generate the raw data files.
if [ ! -f $DIR/raw/lineitem.tbl ] ; then
   mkdir -p $DIR/raw
   cd $DIR/dbgen
   ./dbgen -s 1 -f
   mv *.tbl $DIR/raw
   cd -
fi
