#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../conf.sh

DATASETS=(bike covtype genhist_set1 genhist_set2 power protein)
QUERIES=3000

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
    echo -e "\tdv ..."
	python $DIR/query_generator.py			    \
	  --dbname=$PGDATABASE --table=$table		\
	  --queries=$QUERIES --selectivity=0.01		\
	  --mcenter=Data --mrange=Volume		    \
	  --out=$dataset/queries/${table}_dv_0.01.sql
	# Generate the uv query set (uniform centers, target volume).
    echo -e "\tuv ..."
    python $DIR/query_generator.py			    \
	  --dbname=$PGDATABASE --table=$table		\
	  --queries=$QUERIES --selectivity=0.01		\
	  --mcenter=Uniform --mrange=Volume		    \
	  --out=$dataset/queries/${table}_uv_0.01.sql
	# Generate the dt query set (data centered, target selectivity).
    echo -e "\tdt ..."
    python $DIR/query_generator.py			    \
	  --dbname=$PGDATABASE --table=$table		\
	  --queries=$QUERIES --selectivity=0.01		\
	  --mcenter=Data --mrange=Tuples		    \
	  --out=$dataset/queries/${table}_dt_0.01.sql
	## Generate the ut query set (uniform centers, target selectivity).
    echo -e "\tut ..."
    python $DIR/query_generator.py			    \
	  --dbname=$PGDATABASE --table=$table		\
	  --queries=$QUERIES --selectivity=0.01		\
	  --mcenter=Uniform --mrange=Tuples		    \
	  --out=$dataset/queries/${table}_ut_0.01.sql
  done
done
