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

#<Data, V[1%]>
dv_tables=("forest4")
#<Uniform, V[1%]>
uv_tables=("forest4")
#<Gauss, V[1]>
gv_tables=("forest4")

#<Data, T[1%]>
dt_tables=("forest4")

#<Uniform, T[1%]>
ut_tables=("forest4")

#<Gauss, T[1%]>
gt_tables=("forest4")

workload_5_tables=()


for table in "${dv_tables[@]}"
do
	echo $table
	python $BASEDIR/../stholes.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=10000 \
		--selectivity=0.01 --mcenter=Data --mrange=Volume	\
		--out=$BASEDIR/queries/${table}_dv.sql &
done

for table in "${uv_tables[@]}"
do
	echo $table
	python $BASEDIR/../stholes.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=10000 \
		--selectivity=0.01 --mcenter=Uniform --mrange=Volume	\
		--out=$BASEDIR/queries/${table}_uv.sql &
done

for table in "${gv_tables[@]}"
do
	echo $table
	python $BASEDIR/../stholes.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=10000 \
		--selectivity=0.01 --mcenter=Gauss --clusters=100 --sigma=25 --mrange=Volume	\
		--out=$BASEDIR/queries/${table}_gv.sql &
done


for table in "${dt_tables[@]}"
do
	echo $table
	python $BASEDIR/../stholes.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=10000 \
		--selectivity=0.01 --tolerance=0.002 --mcenter=Data --mrange=Tuples	\
		--out=$BASEDIR/queries/${table}_dt.sql &
done

for table in "${ut_tables[@]}"
do
	echo $table
	python $BASEDIR/../stholes.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=10000 \
		--selectivity=0.01 --tolerance=0.002 --mcenter=Uniform --mrange=Tuples	\
		--out=$BASEDIR/queries/${table}_ut.sql &
done

for table in "${gt_tables[@]}"
do
	echo $table
	python $BASEDIR/../stholes.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=10000 \
		--selectivity=0.01 --tolerance=0.002 --mcenter=Gauss --clusters=100 --sigma=25 --mrange=Tuples	\
		--out=$BASEDIR/queries/${table}_gt.sql &
done

wait

for table in "${workload_5_tables[@]}"
do
	# Workload 5: Random range queries.
	echo $table
	python $BASEDIR/../../genhist/generateRandomWorkload.py \
		--dbname=$DATABASE --database=$DBMS --table=$table --queries=10000 \
		--type=range --out=$BASEDIR/queries/${table}_5.sql
done
