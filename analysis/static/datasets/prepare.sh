#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../conf.sh

DATASETS=(bike genhist_set1 power protein forest)
QUERIES=2500

for dataset in "${DATASETS[@]}" ; do
  # Download, prepare and load the dataset.
  $DIR/$dataset/download.sh
  $DIR/$dataset/load.sh
  mkdir -p $DIR/$dataset/data
  mkdir -p $DIR/$dataset/queries
  source $DIR/$dataset/tables.sh
  # Now build the queries.
  for table in "${TABLES[@]}"; do
    # Extract the table so the python script can impor them into sqlite.
    echo "COPY $table TO '$DIR/$dataset/data/$table.csv' DELIMITER '|';" | $PSQL $PGDATABASE
    # Generate the dv query set (data centered, target volume).
    echo "Generating DV workload for table $table ... "
    $PYTHON $DIR/query_generator.py                          \
     --data=$DIR/$dataset/data/$table.csv --queries=$QUERIES \
     --selectivity=0.01 --mcenter=Data                       \
     --mrange=Volume                                         \
     --out=$DIR/$dataset/queries/${table}_dv_0.01.csv &
    DVPID=$!
    # Generate the uv query set (uniform centers, target volume).
    echo "Generating UV workload for table $table ... "
    $PYTHON $DIR/query_generator.py                          \
     --data=$DIR/$dataset/data/$table.csv --queries=$QUERIES \
     --selectivity=0.01 --mcenter=Uniform                    \
     --mrange=Volume                                         \
     --out=$DIR/$dataset/queries/${table}_uv_0.01.csv &
    UVPID=$!
    # Generate the dt query set (data centered, target selectivity).
    echo "Generating DT workload for table $table ... "
    $PYTHON $DIR/query_generator.py                          \
     --data=$DIR/$dataset/data/$table.csv --queries=$QUERIES \
     --selectivity=0.01 --mcenter=Data                       \
     --mrange=Tuples                                         \
     --out=$DIR/$dataset/queries/${table}_dt_0.01.csv &
    DTPID=$!
    # Generate the ut query set (uniform centers, target selectivity).
    echo "Generating UT workload for table $table ... "
    $PYTHON $DIR/query_generator.py                          \
     --data=$DIR/$dataset/data/$table.csv --queries=$QUERIES \
     --selectivity=0.01 --mcenter=Uniform                    \
     --mrange=Tuples                                         \
     --out=$DIR/$dataset/queries/${table}_ut_0.01.csv &
    UTPID=$!
    # Wait for the query generation:
    wait $DVPID
    wait $UVPID
    wait $DTPID
    wait $UTPID
    # And convert the CSV into SQL files.
    $PYTHON $DIR/query_formatter.py                         \
     --queries=$DIR/$dataset/queries/${table}_dv_0.01.csv   \
     --table=$table                                         \
     --out=$DIR/$dataset/queries/${table}_dv_0.01.sql 
    $PYTHON $DIR/query_formatter.py                         \
     --queries=$DIR/$dataset/queries/${table}_uv_0.01.csv   \
     --table=$table                                         \
     --out=$DIR/$dataset/queries/${table}_uv_0.01.sql 
    $PYTHON $DIR/query_formatter.py                         \
     --queries=$DIR/$dataset/queries/${table}_dt_0.01.csv   \
     --table=$table                                         \
     --out=$DIR/$dataset/queries/${table}_dt_0.01.sql 
    $PYTHON $DIR/query_formatter.py                         \
     --queries=$DIR/$dataset/queries/${table}_ut_0.01.csv   \
     --table=$table                                         \
     --out=$DIR/$dataset/queries/${table}_ut_0.01.sql 
  done
done

# Finally, create the scaled datasets and query workloads.
#$PYTHON $DIR/scaleDatasets.py --dbname=$PGDATABASE --port=$PGPORT
#for dataset in "${DATASETS[@]}" ; do
#  source $dataset/tables.sh
#  for table in "${TABLES[@]}"; do
#    cd $dataset/queries
#    $PYTHON $DIR/scaleExperiments.py                 \
#      --dbname=$PGDATABASE --port=$PGPORT           #\
#      --queryfile=${table}_ut_0.01.sql
#    $PYTHON $DIR/scaleExperiments.py                 \
#      --dbname=$PGDATABASE --port=$PGPORT           \
#      --queryfile=${table}_uv_0.01.sql
#    $PYTHON $DIR/scaleExperiments.py                 \
#      --dbname=$PGDATABASE --port=$PGPORT           \
#      --queryfile=${table}_dt_0.01.sql
#    $PYTHON $DIR/scaleExperiments.py                 \
#      --dbname=$PGDATABASE --port=$PGPORT           \
#      --queryfile=${table}_dv_0.01.sql
#    cd -
#  done
#done
