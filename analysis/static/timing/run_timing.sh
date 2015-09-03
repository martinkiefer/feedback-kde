#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../conf.sh

# Some general parameters.
REPETITIONS=4
QUERIES=50

DIMENSIONS=(8)
MODELSIZES=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152)

LOGPREFIX=$DIR/../../evaluation/timing/

for dimension in "${DIMENSIONS[@]}"; do
    echo "Running for $dimension dimensions:"
    for modelsize in "${MODELSIZES[@]}"; do
      echo "  Running with modelsize $modelsize:"
      stholes_modelsize=$((modelsize / 2))
      $POSTGRES -D $PGDATAFOLDER -p $PGPORT >>  postgres.log 2>&1 &
      PGPID=$!
      sleep 2
      for i in $(seq 1 $REPETITIONS); do
         echo "    Repetition $i of $REPETITIONS:"
            
         echo "      KDE Heuristic (CPU):"
         $PYTHON $DIR/runTimingExperiment.py                    \
            --dbname=$PGDATABASE --port=$PGPORT                \
            --dimensions=$dimension --log=$LOGPREFIX/result_time.csv \
            --model=kde_heuristic --modelsize=$modelsize       \
            --trainqueries=0 --queries=$QUERIES 
           
         echo "      KDE Adaptive (CPU):"
         $PYTHON $DIR/runTimingExperiment.py                    \
            --dbname=$PGDATABASE --port=$PGPORT                \
            --dimensions=$dimension --log=$LOGPREFIX/result_time.csv \
            --model=kde_adaptive --modelsize=$modelsize        \
            --trainqueries=0 --queries=$QUERIES   
            
         echo "      KDE Heuristic (GPU):"
         $PYTHON $DIR/runTimingExperiment.py                    \
            --dbname=$PGDATABASE --port=$PGPORT                \
            --dimensions=$dimension --log=$LOGPREFIX/result_time.csv \
            --model=kde_heuristic --modelsize=$modelsize       \
            --trainqueries=0 --queries=$QUERIES --gpu
           
         echo "      KDE Adaptive (GPU):"
         $PYTHON $DIR/runTimingExperiment.py                    \
            --dbname=$PGDATABASE --port=$PGPORT                \
            --dimensions=$dimension --log=$LOGPREFIX/result_time.csv \
            --model=kde_adaptive --modelsize=$modelsize        \
            --trainqueries=0 --queries=$QUERIES --gpu       
      done
      kill -9 $PGPID
      sleep 5
    done
done
