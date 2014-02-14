#/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

cd /tmp
python $BASEDIR/generator1.py
cd -

#First drop the existing tables
$BASEDIR/drop-set1-tables.sh

#PSQL command
$PSQL $DATABASE $USER << EOF
	CREATE TABLE gen1_d3(
		c1 double precision,
		c2 double precision,
		c3 double precision);
	COPY gen1_d3 FROM '/tmp/data_gen1_d3.csv' DELIMITER',';
	CREATE TABLE gen1_d4(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision);
	COPY gen1_d4 FROM '/tmp/data_gen1_d4.csv' DELIMITER',';
	CREATE TABLE gen1_d5(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision);
	COPY gen1_d5 FROM '/tmp/data_gen1_d5.csv' DELIMITER',';
	CREATE TABLE gen1_d8(
		c1 double precision,
		c2 double precision,
		c3 double precision,
		c4 double precision,
		c5 double precision,
		c6 double precision,
		c7 double precision,
		c8 double precision);
	COPY gen1_d8 FROM '/tmp/data_gen1_d8.csv' DELIMITER',';
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
	COPY gen1_d10 FROM '/tmp/data_gen1_d10.csv' DELIMITER',';
EOF
