#/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

# Check if the tables folder exists and has all the files. If not, create them.
if [ ! -e $BASEDIR/tables/data_gen1_d5.csv ]; then
	mkdir -p $BASEDIR/tables
	cd $BASEDIR/tables
	python ../generator1.py
	cd -
fi

# First drop the existing tables
$BASEDIR/drop-set1-tables.sh

# PSQL command
$PSQL $PGDATABASE $USER << EOF
	CREATE TABLE gen1_d2(
		c1 double precision,
		c2 double precision);
	COPY gen1_d2 FROM '$BASEDIR/tables/data_gen1_d2.csv' DELIMITER',';
	CREATE TABLE gen1_d3(
		c1 double precision,
		c2 double precision,
		c3 double precision);
	COPY gen1_d3 FROM '$BASEDIR/tables/data_gen1_d3.csv' DELIMITER',';
	CREATE TABLE gen1_d4(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision);
	COPY gen1_d4 FROM '$BASEDIR/tables/data_gen1_d4.csv' DELIMITER',';
	CREATE TABLE gen1_d5(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision);
	COPY gen1_d5 FROM '$BASEDIR/tables/data_gen1_d5.csv' DELIMITER',';
	CREATE TABLE gen1_d8(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision,
		c6 double precision,
		c7 double precision,
		c8 double precision);
	COPY gen1_d8 FROM '$BASEDIR/tables/data_gen1_d8.csv' DELIMITER',';
	CREATE TABLE gen1_d10(
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
	COPY gen1_d10 FROM '$BASEDIR/tables/data_gen1_d10.csv' DELIMITER',';
EOF

# MonetDB command
if [ -z $MONETDATABASE ]; then
	exit
fi

echo "
	CREATE TABLE gen1_d3(
		c1 double precision,
		c2 double precision,
		c3 double precision);
	COPY INTO gen1_d3 FROM '$BASEDIR/tables/data_gen1_d3.csv' USING DELIMITERS ',','\r\n';
		CREATE TABLE gen1_d4(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision);
	COPY INTO gen1_d4 FROM '$BASEDIR/tables/data_gen1_d4.csv' USING DELIMITERS ',','\r\n';
	CREATE TABLE gen1_d5(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision);
	COPY INTO gen1_d5 FROM '$BASEDIR/tables/data_gen1_d5.csv' USING DELIMITERS ',','\r\n';
	CREATE TABLE gen1_d8(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision,
		c6 double precision,
		c7 double precision,
		c8 double precision);
	COPY INTO gen1_d8 FROM '$BASEDIR/tables/data_gen1_d8.csv' USING DELIMITERS ',','\r\n';
	CREATE TABLE gen1_d10(
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
	COPY INTO gen1_d10 FROM '$BASEDIR/tables/data_gen1_d10.csv' USING DELIMITERS ',','\r\n';
" | mclient -lsql -d$MONETDATABASE

