#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../conf.sh

DATASETS=(bike covtype genhist_set1 genhist_set2 power protein)
QUERIES=2500

for dataset in "${DATASETS[@]}" ; do
  # Download, prepare and load the dataset.
  $dataset/download.sh
  $dataset/load.sh
  # Now build the queries.
  mkdir -p $dataset/queries
  source $dataset/tables.sh
  for table in "${TABLES[@]}"; do
    echo "Generating workload for table $table ... "
    # Generate the dv query set (data centered, target volume).
    python $DIR/query_generator.py                  \
     --dbname=$PGDATABASE --table=$table            \
     --queries=$QUERIES --selectivity=0.01          \
     --mcenter=Data --mrange=Volume                 \
     --out=$dataset/queries/${table}_dv_0.01.sql &
    DVPID=$!
    # Generate the uv query set (uniform centers, target volume).
    python $DIR/query_generator.py                 \
     --dbname=$PGDATABASE --table=$table           \
     --queries=$QUERIES --selectivity=0.01         \
     --mcenter=Uniform --mrange=Volume             \
     --out=$dataset/queries/${table}_uv_0.01.sql &
    UVPID=$!
    # Generate the dt query set (data centered, target selectivity).
    python $DIR/query_generator.py                 \
     --dbname=$PGDATABASE --table=$table           \
     --queries=$QUERIES --selectivity=0.01         \
     --mcenter=Data --mrange=Tuples                \
     --out=$dataset/queries/${table}_dt_0.01.sql &
    DTPID=$!
    # Generate the ut query set (uniform centers, target selectivity).
    python $DIR/query_generator.py                 \
     --dbname=$PGDATABASE --table=$table           \
     --queries=$QUERIES --selectivity=0.01         \
     --mcenter=Uniform --mrange=Tuples             \
     --out=$dataset/queries/${table}_ut_0.01.sql &
    UTPID=$!
    # Wait for the query generation:
    wait $DVPID
    wait $UVPID
    wait $DTPID
    wait $UTPID
  done
done

# Finally, create the scaled datasets and query workloads.
python $DIR/scaleExperiments.py --dbname=$PGDATABASE
