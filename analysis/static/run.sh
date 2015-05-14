#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../conf.sh

# Some general parameters.
REPETITIONS=10
TRAINQUERIES=100
TESTQUERIES=300
MODELSIZE=1024
LOGFILE=$DIR/result.csv

# Prepare a new result file.
echo > $LOGFILE

for dataset in $DIR/datasets/*; do
    [ -d "${dataset}" ] || continue # if not a directory, skip
    dataset_name=`basename $dataset`
    echo "Running experiments for $dataset_name:"
    for query in $dataset/queries/*; do
        [ -f "${query}" ] || continue
        query_file=`basename $query`
        echo "  Running for query file $query_file:"
        # Reinitialize postgres (just to be safe)
        postgres -D $PGDATAFOLDER -p $PGPORT > postgres.log 2>&1 &
        PGPID=$!
        sleep 2
        for i in $(seq 1 $REPETITIONS); do
           echo "    Repetition $i:"
           TS=$SECONDS

           # Pick a new experiment and run batch:
           echo "      KDE (batch):"
           python $DIR/runExperiment.py                                 \
              --dbname=$PGDATABASE --port=$PGPORT                       \
              --queryfile=$query --log=$LOGFILE                         \
              --model=kde_batch --modelsize=$MODELSIZE                  \
              --trainqueries=$TRAINQUERIES --testqueries=$TESTQUERIES
            
           # Run KDE heuristic: 
           echo "      KDE (heuristic):"
           python $DIR/runExperiment.py                                 \
              --dbname=$PGDATABASE --port=$PGPORT                       \
              --queryfile=$query --log=$LOGFILE                         \
              --model=kde_heuristic --modelsize=$MODELSIZE              \
              --trainqueries=$TRAINQUERIES --testqueries=$TESTQUERIES   \
              --replay_experiment
            
           # Run KDE optimal: 
           echo "      KDE (SCV):"
           python $DIR/runExperiment.py                                 \
              --dbname=$PGDATABASE --port=$PGPORT                       \
              --queryfile=$query --log=$LOGFILE                         \
              --model=kde_scv --modelsize=$MODELSIZE                    \
              --trainqueries=$TRAINQUERIES --testqueries=$TESTQUERIES   \
              --replay_experiment
            
           # Run KDE adpative: 
           echo "      KDE (adaptive):"
           python $DIR/runExperiment.py                                 \
              --dbname=$PGDATABASE --port=$PGPORT                       \
              --queryfile=$query --log=$LOGFILE                         \
              --model=kde_adaptive --modelsize=$MODELSIZE --logbw       \
              --trainqueries=$TRAINQUERIES --testqueries=$TESTQUERIES   \
              --replay_experiment
           
           # Run stholes:  
           echo "      STHoles:"
           python $DIR/runExperiment.py                                 \
              --dbname=$PGDATABASE --port=$PGPORT                       \
              --queryfile=$query --log=$LOGFILE                         \
              --model=stholes --modelsize=$MODELSIZE                    \
              --trainqueries=$TRAINQUERIES --testqueries=$TESTQUERIES   \
              --replay_experiment

           # Run with Postgres Histograms:
           echo "      Postgres histograms:"
           python $DIR/runExperiment.py                                 \
              --dbname=$PGDATABASE --port=$PGPORT                       \
              --queryfile=$query --log=$LOGFILE                         \
              --model=postgres --modelsize=$MODELSIZE                   \
              --trainqueries=$TRAINQUERIES --testqueries=$TESTQUERIES   \
              --replay_experiment
           
           # Run without statistics:
           echo "      No statistics:"
           python $DIR/runExperiment.py                                 \
              --dbname=$PGDATABASE --port=$PGPORT                       \
              --queryfile=$query --log=$LOGFILE                         \
              --model=none --modelsize=$MODELSIZE                       \
              --trainqueries=$TRAINQUERIES --testqueries=$TESTQUERIES   \
              --replay_experiment
           
           ELAPSED=$(($SECONDS - $TS))
           echo "    Repetition finished (took $ELAPSED seconds)!"

        done
        kill -9 $PGPID
        sleep 2
    done
done
