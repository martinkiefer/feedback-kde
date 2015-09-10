#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../conf.sh

# Some general parameters.
REPETITIONS=10
TRAINQUERIES=100
TESTQUERIES=100
LOGFILE=$DIR/../evaluation/model_size/result.csv

dataset=$DIR/datasets/forest
query=$dataset/queries/forest8_dt_0.01.sql

MODELSIZES=(512 1024 2048 4096 8192 16384 32768 65536)

# Prepare a new result file.
#echo > $LOGFILE

for MODELSIZE in "${MODELSIZES[@]}"; do
        query_file=`basename $query`
        echo "  Running for query file $query_file:"
        # Reinitialize postgres (just to be safe)
        $POSTGRES -D $PGDATAFOLDER -p $PGPORT >>  postgres.log 2>&1 &
        PGPID=$!
        sleep 2
        for i in $(seq 1 $REPETITIONS); do
           echo "    Repetition $i:"
           TS=$SECONDS

           echo "      KDE (batch):"
           $PYTHON $DIR/runExperiment.py                                 \
              --dbname=$PGDATABASE --port=$PGPORT                       \
              --queryfile=$query --log=$LOGFILE                         \
              --model=kde_batch --modelsize=$MODELSIZE                  \
              --trainqueries=$TRAINQUERIES --testqueries=$TESTQUERIES

           # Run KDE heuristic: 
           echo "      KDE (heuristic):"
           $PYTHON $DIR/runExperiment.py                                 \
              --dbname=$PGDATABASE --port=$PGPORT                       \
              --queryfile=$query --log=$LOGFILE                         \
              --model=kde_heuristic --modelsize=$MODELSIZE              \
              --trainqueries=$TRAINQUERIES --testqueries=$TESTQUERIES   \
              --replay_experiment

           # Run KDE adpative: 
           echo "      KDE (adaptive):"
           $PYTHON $DIR/runExperiment.py                                 \
              --dbname=$PGDATABASE --port=$PGPORT                       \
              --queryfile=$query --log=$LOGFILE                         \
              --model=kde_adaptive --modelsize=$MODELSIZE --logbw       \
              --trainqueries=$TRAINQUERIES --testqueries=$TESTQUERIES   \
              --replay_experiment

           ELAPSED=$(($SECONDS - $TS))
           echo "    Repetition finished (took $ELAPSED seconds)!"

        done
        kill -9 $PGPID
        sleep 2
    done
