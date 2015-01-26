#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../conf.sh

# Some general parameters.
REPETITIONS=2
QUERIES=25

#DIMENSIONS=(5 8)
#MODELSIZES=(4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152)

DIMENSIONS=(8)
MODELSIZES=(4096)

# Prepare a new result file.
echo > $DIR/result_time.csv

for dimension in "${DIMENSIONS[@]}"; do
    echo "Running for $dimension dimensions:"
    for modelsize in "${MODELSIZES[@]}"; do
      echo "  Running with modelsize $modelsize:"
      stholes_modelsize=$((modelsize / 2))
      postgres -D $PGDATAFOLDER -p $PGPORT >> /tmp/postgres.log 2>&1 &
      PGPID=$!
      sleep 2
      for i in $(seq 1 $REPETITIONS); do
         echo "    Repetition $i:"
            
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
