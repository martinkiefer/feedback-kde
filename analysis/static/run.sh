#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../conf.sh

# Some general parameters.
REPETITIONS=50
TRAINQUERIES=100
QUERIES=300
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
        # Reinitialize postgres (just to be safe)
        postgres -D $PGDATAFOLDER -p $PGPORT > /dev/null 2>&1 &
        PGPID=$!
        sleep 2
        
        for i in $(seq 1 $REPETITIONS); do
           # Pick a new experiment (and run batch).
           python $DIR/runExperiment.py                        \
              --dbname=$PGDATABASE --port=$PGPORT              \
              --queryfile=$query --log=$DIR/result.csv         \
              --model=kde_batch --modelsize=$KDE_MODELSIZE     \
              --trainqueries=$TRAINQUERIES --queries=$QUERIES  \
              --error=relative --record

            # Run stholes: 
            python $DIR/replayExperiment.py                    \
              --dbname=$PGDATABASE --port=$PGPORT              \
              --model=stholes --modelsize=$STHOLES_MODELSIZE   \
              --error=relative --log=$DIR/result.csv
            
            # Run KDE heuristic: 
            python $DIR/replayExperiment.py                    \
              --dbname=$PGDATABASE --port=$PGPORT              \
              --model=kde_heuristic                            \
              --error=relative --log=$DIR/result.csv
            
            # Run KDE optimal: 
            python $DIR/replayExperiment.py                    \
              --dbname=$PGDATABASE --port=$PGPORT              \
              --model=kde_optimal                              \
              --error=relative --log=$DIR/result.csv
            
            # Run KDE adpative: 
            python $DIR/replayExperiment.py                    \
              --dbname=$PGDATABASE --port=$PGPORT              \
              --model=kde_adaptive_rmsprop                     \
              --error=relative --log=$DIR/result.csv
            
        done
        kill -9 $PGPID
        sleep 2
    done
done
