#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../conf.sh

# Some general parameters.
REPETITIONS=50
TRAINQUERIES=100
QUERIES=100
STHOLES_MODELSIZE=512
KDE_MODELSIZE=1024

# Prepare a new result file.
echo > $DIR/result.csv

for dataset in $DIR/*; do
    [ -d "${dataset}" ] || continue # if not a directory, skip
    datset_name=`basename $dataset`
    echo "Running experiments for $datset_name:"
    for query in $dataset/queries/*; do
        [ -f "${query}" ] || continue
        query_file=`basename $query`
        echo -e "\tRunning query $query_file."
        for i in $(seq 1 $REPETITIONS); do
            postgres -D $PGDATAFOLDER -p $PGPORT >/dev/null 2>&1 &
            PGPID=$!
            sleep 2
            python $DIR/runExperiment.py                         \
               --dbname=$PGDATABASE --port=$PGPORT               \
               --queryfile=$query --log=$DIR/result.csv          \
               --model=stholes --modelsize=$STHOLES_MODELSIZE    \
               --trainqueries=$TRAINQUERIES --queries=$QUERIES   \
               --error=relative

            python $DIR/runExperiment.py                         \
               --dbname=$PGDATABASE --port=$PGPORT               \
               --queryfile=$query --log=$DIR/result.csv          \
               --model=kde_heuristic --modelsize=$KDE_MODELSIZE  \
               --trainqueries=$TRAINQUERIES --queries=$QUERIES   \
               --error=relative

            python $DIR/runExperiment.py                         \
               --dbname=$PGDATABASE --port=$PGPORT               \
               --queryfile=$query --log=$DIR/result.csv          \
               --model=kde_optimal --modelsize=$KDE_MODELSIZE    \
               --trainqueries=$TRAINQUERIES --queries=$QUERIES   \
               --error=relative --reuse

            python $DIR/runExperiment.py                         \
               --dbname=$PGDATABASE --port=$PGPORT               \
               --queryfile=$query --log=$DIR/result.csv          \
               --model=kde_batch --modelsize=$KDE_MODELSIZE      \
               --trainqueries=$TRAINQUERIES --queries=$QUERIES   \
               --error=relative --reuse
            
            kill -9 $PGPID
            sleep 2
        done
    done
done
