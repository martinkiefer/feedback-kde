#/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

mkdir -p $BASEDIR/queries
mkdir -p $BASEDIR/tables
#
#Create some tables
for i in 3 4 5 8 10
do
  if [ ! -f $BASEDIR/tables/data_mvt_d$i.csv ] 
  then
    python $BASEDIR/../movingTarget.py --table mvt --dataoutput $BASEDIR/tables/data_mvt_d$i.csv --queryoutput $BASEDIR/queries/mvt_d$i.sql \
    --sigma 0.01 --margin 0.4 --clusters 10 --points 100 --steps 10 --dimensions $i --queriesperstep 10 --maxprob 0.9 
  fi
done
# First drop the existing tables
$BASEDIR/drop-mvt-tables.sh

# PSQL command
$PSQL $PGDATABASE $USER << EOF
	CREATE TABLE mvt_d3(
		c1 double precision,
		c2 double precision,
		c3 double precision);
	COPY mvt_d3 FROM '$BASEDIR/tables/data_mvt_d3.csv' DELIMITER',';
	CREATE TABLE mvt_d4(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision);
	COPY mvt_d4 FROM '$BASEDIR/tables/data_mvt_d4.csv' DELIMITER',';
	CREATE TABLE mvt_d5(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision);
	COPY mvt_d5 FROM '$BASEDIR/tables/data_mvt_d5.csv' DELIMITER',';
	CREATE TABLE mvt_d8(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision,
		c6 double precision,
		c7 double precision,
		c8 double precision);
	COPY mvt_d8 FROM '$BASEDIR/tables/data_mvt_d8.csv' DELIMITER',';
	CREATE TABLE mvt_d10(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision,
		c6 double precision,
		c7 double precision,
		c8 double precision,
		c9 double precision,
		c10 double precision);
	COPY mvt_d10 FROM '$BASEDIR/tables/data_mvt_d10.csv' DELIMITER',';
EOF

# MonetDB command
if [ -z $MONETDATABASE ]; then
	exit
fi

echo "
	CREATE TABLE mvt_d3(
		c1 double precision,
		c2 double precision,
		c3 double precision);
	COPY INTO mvt_d3 FROM '$BASEDIR/tables/data_mvt_d3.csv' USING DELIMITERS ',','\r\n';
		CREATE TABLE mvt_d4(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision);
	COPY INTO mvt_d4 FROM '$BASEDIR/tables/data_mvt_d4.csv' USING DELIMITERS ',','\r\n';
	CREATE TABLE mvt_d5(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision);
	COPY INTO mvt_d5 FROM '$BASEDIR/tables/data_mvt_d5.csv' USING DELIMITERS ',','\r\n';
	CREATE TABLE mvt_d8(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision,
		c6 double precision,
		c7 double precision,
		c8 double precision);
	COPY INTO mvt_d8 FROM '$BASEDIR/tables/data_mvt_d8.csv' USING DELIMITERS ',','\r\n';
	CREATE TABLE mvt_d10(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision,
		c6 double precision,
		c7 double precision,
		c8 double precision,
		c9 double precision,
		c10 double precision);
	COPY INTO mvt_d10 FROM '$BASEDIR/tables/data_mvt_d10.csv' USING DELIMITERS ',','\r\n';
" | mclient -lsql -d$MONETDATABASE

