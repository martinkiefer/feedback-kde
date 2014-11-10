#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../conf.sh

# Some general parameters.
REPETITIONS=10
TRAINQUERIES=100
QUERIES=1000

MODELSIZES=(128 256 512 1024 2048 4096 8192 16384 32768)

# Prepare a new result file.
echo > $DIR/result_time.csv

for dataset in $DIR/*; do
    [ -d "${dataset}" ] || continue # if not a directory, skip
    datset_name=`basename $dataset`
    echo "Running experiments for $datset_name:"
    for query in $dataset/queries/*; do
      [ -f "${query}" ] || continue
      for modelsize in "${MODELSIZES[@]}"; do
        query_file=`basename $query`
        echo -e "\tRunning query $query_file."
        postgres -D $PGDATAFOLDER -p $PGPORT > /dev/null 2>&1 &
        PGPID=$!
        sleep 2
        for i in $(seq 1 $REPETITIONS); do
            python $DIR/runTimingExperiment.py                   \
               --dbname=$PGDATABASE --port=$PGPORT               \
               --queryfile=$query --log=$DIR/result_time.csv     \
               --model=none --modelsize=$modelsize               \
               --trainqueries=$TRAINQUERIES --queries=$QUERIES   

            python $DIR/runTimingExperiment.py                   \
               --dbname=$PGDATABASE --port=$PGPORT               \
               --queryfile=$query --log=$DIR/result_time.csv     \
               --model=kde_heuristic --modelsize=$modelsize      \
               --trainqueries=$TRAINQUERIES --queries=$QUERIES
           
            python $DIR/runTimingExperiment.py                   \
               --dbname=$PGDATABASE --port=$PGPORT               \
               --queryfile=$query --log=$DIR/result_time.csv     \
               --model=kde_adaptive --modelsize=$modelsize       \
               --trainqueries=$TRAINQUERIES --queries=$QUERIES   
            
            python $DIR/runTimingExperiment.py                   \
               --dbname=$PGDATABASE --port=$PGPORT               \
               --queryfile=$query --log=$DIR/result_time.csv     \
               --model=kde_batch --modelsize=$modelsize          \
               --trainqueries=$TRAINQUERIES --queries=$QUERIES 
            
            python $DIR/runTimingExperiment.py                   \
               --dbname=$PGDATABASE --port=$PGPORT               \
               --queryfile=$query --log=$DIR/result_time.csv     \
               --model=kde_heuristic --modelsize=$modelsize      \
               --trainqueries=$TRAINQUERIES --queries=$QUERIES --gpu
           
            python $DIR/runTimingExperiment.py                   \
               --dbname=$PGDATABASE --port=$PGPORT               \
               --queryfile=$query --log=$DIR/result_time.csv     \
               --model=kde_adaptive --modelsize=$modelsize       \
               --trainqueries=$TRAINQUERIES --queries=$QUERIES --gpu       
            
            python $DIR/runTimingExperiment.py                   \
               --dbname=$PGDATABASE --port=$PGPORT               \
               --queryfile=$query --log=$DIR/result_time.csv     \
               --model=kde_batch --modelsize=$modelsize          \
               --trainqueries=$TRAINQUERIES --queries=$QUERIES --gpu
        done
        kill -9 $PGPID
        sleep 2
      done
    done
done
