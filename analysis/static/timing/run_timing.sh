#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../conf.sh

# Some general parameters.
REPETITIONS=4
QUERIES=50

DIMENSIONS=(2 3 4 5 6 7 8)
MODELSIZES=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152)

# Prepare a new result file.
echo > $DIR/result_dim.csv

for dimension in "${DIMENSIONS[@]}"; do
    echo "Running for $dimension dimensions:"
    for modelsize in "${MODELSIZES[@]}"; do
      echo "  Running with modelsize $modelsize:"
      stholes_modelsize=$((modelsize / 2))
      postgres -D $PGDATAFOLDER -p $PGPORT >> postgres.log 2>&1 &
      PGPID=$!
      sleep 2
      for i in $(seq 1 $REPETITIONS); do
         echo "    Repetition $i of $REPETITIONS:"
            
         echo "      STHoles:"
         python $DIR/runTimingExperiment.py                    \
            --dbname=$PGDATABASE --port=$PGPORT                \
            --dimensions=$dimension --log=$DIR/result_time.csv \
            --model=stholes --modelsize=$stholes_modelsize     \
            --trainqueries=0 --queries=$QUERIES 
         
         echo "      KDE Heuristic (CPU):"
         python $DIR/runTimingExperiment.py                    \
            --dbname=$PGDATABASE --port=$PGPORT                \
            --dimensions=$dimension --log=$DIR/result_time.csv \
            --model=kde_heuristic --modelsize=$modelsize       \
            --trainqueries=0 --queries=$QUERIES 
           
         echo "      KDE Adaptive (CPU):"
         python $DIR/runTimingExperiment.py                    \
            --dbname=$PGDATABASE --port=$PGPORT                \
            --dimensions=$dimension --log=$DIR/result_time.csv \
            --model=kde_adaptive --modelsize=$modelsize        \
            --trainqueries=0 --queries=$QUERIES   
            
         echo "      KDE Heuristic (GPU):"
         python $DIR/runTimingExperiment.py                    \
            --dbname=$PGDATABASE --port=$PGPORT                \
            --dimensions=$dimension --log=$DIR/result_time.csv \
            --model=kde_heuristic --modelsize=$modelsize       \
            --trainqueries=0 --queries=$QUERIES --gpu
           
         echo "      KDE Adaptive (GPU):"
         python $DIR/runTimingExperiment.py                    \
            --dbname=$PGDATABASE --port=$PGPORT                \
            --dimensions=$dimension --log=$DIR/result_time.csv \
            --model=kde_adaptive --modelsize=$modelsize        \
            --trainqueries=0 --queries=$QUERIES --gpu       
      done
      kill -9 $PGPID
      sleep 2
    done
done
