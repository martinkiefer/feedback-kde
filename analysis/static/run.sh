#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../conf.sh

# Some general parameters.
REPETITIONS=10
TRAINQUERIES=100
QUERIES=300
STHOLES_MODELSIZE=512
KDE_MODELSIZE=1024
ERROR=absolute
LOGFILE=$DIR/result_new.csv

# Prepare a new result file.
echo > $LOGFILE

for dataset in $DIR/datasets/*; do
    [ -d "${dataset}" ] || continue # if not a directory, skip
    dataset_name=`basename $dataset`
    echo "Running experiments for $dataset_name:"
    for query in $dataset/queries/*; do
        [ -f "${query}" ] || continue
        query_file=`basename $query`
        echo "  Running query $query_file:"
        # Reinitialize postgres (just to be safe)
        $POSTGRES -D $PGDATAFOLDER -p $PGPORT > postgres.log 2>&1 &
        PGPID=$!
        sleep 2
        for i in $(seq 1 $REPETITIONS); do
           echo "    Repetition $i:"

           # Pick a new experiment (and run batch).
           echo "      KDE batch:"
           $PYTHON $DIR/runExperiment.py                        \
              --dbname=$PGDATABASE --port=$PGPORT              \
              --queryfile=$query --log=$LOGFILE                \
              --model=kde_batch --modelsize=$KDE_MODELSIZE     \
              --trainqueries=$TRAINQUERIES --queries=$QUERIES  \
              --error=$ERROR --record

           # Run stholes:  
           echo "      STHoles:"
           $PYTHON $DIR/replayExperiment.py                     \
              --dbname=$PGDATABASE --port=$PGPORT              \
              --model=stholes --modelsize=$STHOLES_MODELSIZE   \
              --error=$ERROR --log=$LOGFILE
            
           # Run KDE heuristic: 
           echo "      KDE heuristic:"
           $PYTHON $DIR/replayExperiment.py                     \
              --dbname=$PGDATABASE --port=$PGPORT              \
              --model=kde_heuristic                            \
              --error=$ERROR --log=$LOGFILE
            
           # Run KDE optimal: 
           echo "      KDE optimal:"
           $PYTHON $DIR/replayExperiment.py                     \
              --dbname=$PGDATABASE --port=$PGPORT              \
              --model=kde_optimal                              \
              --error=$ERROR --log=$LOGFILE
            
           # Run KDE adpative: 
           echo "      KDE adaptive:"
           $PYTHON $DIR/replayExperiment.py                     \
              --dbname=$PGDATABASE --port=$PGPORT              \
              --model=kde_adaptive_rmsprop                     \
              --error=$ERROR --log=$LOGFILE
           
           # Run KDE adpative (log-scaled): 
           echo "      KDE adaptive (log-scaled):"
           $PYTHON $DIR/replayExperiment.py                     \
              --dbname=$PGDATABASE --port=$PGPORT              \
              --model=kde_adaptive_rmsprop --logbw             \
              --error=$ERROR --log=$LOGFILE
           
        done
        kill -9 $PGPID
        sleep 2
    done
done
