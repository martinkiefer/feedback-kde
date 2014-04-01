#/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

mkdir -p $BASEDIR/queries
mkdir -p $BASEDIR/tables
#
#Create some tables
for i in 3 4 5 8 10
do
  python $BASEDIR/../MovingTargetChangingData.py --table mvtc --queryoutput $BASEDIR/queries/mvtc_d$i.sql --history 4 --sigma 0.01 \
    --margin 0.4 --clusters 10 --points 1000 --steps 10 --dimensions $i --queriesperstep 10 --maxprob 0.9 \
    --c1_queries 100 --c1_output $BASEDIR/queries/mvtc_d$i\_O.sql --dataoutput tables/data_mvtc_d$i.csv
done
# First drop the existing tables
$BASEDIR/drop-mvtc-tables.sh

# PSQL command
$PSQL $PGDATABASE $USER << EOF
	CREATE TABLE mvtc_d3(
		CL integer,
		c1 double precision,
		c2 double precision,
		c3 double precision);
	COPY mvtc_d3 FROM '$BASEDIR/tables/data_mvtc_d3.csv' DELIMITER',';
	CREATE TABLE mvtc_d4(
		CL integer,
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision);
	COPY mvtc_d4 FROM '$BASEDIR/tables/data_mvtc_d4.csv' DELIMITER',';
	CREATE TABLE mvtc_d5(
		CL integer,
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision);
	COPY mvtc_d5 FROM '$BASEDIR/tables/data_mvtc_d5.csv' DELIMITER',';
	CREATE TABLE mvtc_d8(
		CL integer,
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision,
		c6 double precision,
		c7 double precision,
		c8 double precision);
	COPY mvtc_d8 FROM '$BASEDIR/tables/data_mvtc_d8.csv' DELIMITER',';
	CREATE TABLE mvtc_d10(
		CL integer,
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
	COPY mvtc_d10 FROM '$BASEDIR/tables/data_mvtc_d10.csv' DELIMITER',';
EOF

# MonetDB command
if [ -z $MONETDATABASE ]; then
	exit
fi

echo "
	CREATE TABLE mvtc_d3(
		CL integer,
		c1 double precision,
		c2 double precision,
		c3 double precision);
	COPY INTO mvtc_d3 FROM '$BASEDIR/tables/data_mvtc_d3.csv' USING DELIMITERS ',','\r\n';
		CREATE TABLE mvtc_d4(
		CL integer,
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision);
	COPY INTO mvtc_d4 FROM '$BASEDIR/tables/data_mvtc_d4.csv' USING DELIMITERS ',','\r\n';
	CREATE TABLE mvtc_d5(
		CL integer,
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision);
	COPY INTO mvtc_d5 FROM '$BASEDIR/tables/data_mvtc_d5.csv' USING DELIMITERS ',','\r\n';
	CREATE TABLE mvtc_d8(
		CL integer,
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision,
		c6 double precision,
		c7 double precision,
		c8 double precision);
	COPY INTO mvtc_d8 FROM '$BASEDIR/tables/data_mvtc_d8.csv' USING DELIMITERS ',','\r\n';
	CREATE TABLE mvtc_d10(
		CL integer,
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
	COPY INTO mvtc_d10 FROM '$BASEDIR/tables/data_mvtc_d10.csv' USING DELIMITERS ',','\r\n';
" | mclient -lsql -d$MONETDATABASE

