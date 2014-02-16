#/usr/bin/bash
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

if [ ! -z $MONETDATABASE ]; then
	DBMS="monetdb"
	DATABASE=$MONETDATABASE
else
	DBMS="postgres"
	DATABASE=$PGDATABASE
fi

mkdir -p $BASEDIR/queries

# Create the different query workloads.

workload_1_tables=("gen2_d5")
workload_2_tables=("gen2_d5")
workload_3_tables=("gen2_d5")
workload_4_tables=()
workload_5_tables=("gen2_d5")

for table in "${workload_1_tables[@]}"
do
	# Workload 1: Random queries with 10% target selectivity.
	python $BASEDIR/../generateWorkloadWithTargetSelectivity.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=10000 \
		--selectivity=0.1 --tolerance=0.05 --method=binary	\
		--out=$BASEDIR/queries/${table}_1.sql
done

for table in "${workload_2_tables[@]}"
do
	# Workload 2: Random queries with 1% target selectivity.
	python $BASEDIR/../generateWorkloadWithTargetSelectivity.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=10000 \
		--selectivity=0.01 --tolerance=0.01 --method=binary	\
		--out=$BASEDIR/queries/${table}_2.sql
done

for table in "${workload_3_tables[@]}"
do
	# Workload 3: Random half-range queries.
	python $BASEDIR/../generateRandomWorkload.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=20000 \
		--type=half_range --out=$BASEDIR/queries/${table}_3.sql
done

for table in "${workload_4_tables[@]}"
do
	# Workload 4: Random queries with 1% target selectivity and 2 random outprojected queries.
	python $BASEDIR/../generateWorkloadWithTargetSelectivity.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=100 \
		--selectivity=0.01 --tolerance=0.01 --method=binary	--fixed=2\
		--out=$BASEDIR/queries/${table}_4.sql
done

for table in "${workload_5_tables[@]}"
do
	# Workload 5: Random range queries.
	python $BASEDIR/../generateRandomWorkload.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=10000 \
		--type=range --out=$BASEDIR/queries/${table}_5.sql
done