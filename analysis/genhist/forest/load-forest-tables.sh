#/usr/bin/bash
#Creates forest cover testdata in postgresql
#Needs information about postgresql and the file containing the forest cover dataset

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

#If the Covtype file is not set, pull it from the website.
if [ ! -e $BASEDIR/raw/covtype.csv ]; then
		mkdir -p $BASEDIR/raw
		cd $BASEDIR/raw
		wget http://kdd.ics.uci.edu/databases/covertype/covtype.data.gz
		gunzip -f covtype.data.gz
		# Extract first 10 columns
		cat covtype.data | cut -d , -f 1-10 > covtype.csv
		cd -
fi

# Drop the existing tables.
$BASEDIR/drop-forest-tables.sh

#Put in temporary table and create permanent tables with normalized values
$PSQL $PGDATABASE $USER << EOF
BEGIN;
CREATE TABLE forest10
(
 c1 double precision, 
 c2 double precision, 
 c3 double precision, 
 c4 double precision, 
 c5 double precision, 
 c6 double precision, 
 c7 double precision, 
 c8 double precision, 
 c9 double precision, 
 c10 double precision
);
copy forest10 from '$BASEDIR/raw/covtype.csv' DELIMITER',' CSV;

SELECT
 c1 AS c1, c5 AS c2, c9 AS c3
 into forest3
 from forest10;

SELECT
 c2 AS c1, c3 AS c2, c7 AS c3, c9 AS c4
 into forest4
 from forest10;

SELECT
 c2 AS c1, c4 AS c2, c5 AS c3, c6 as c4, c8 AS c5
 into forest5
 from forest10;

SELECT
 c1 AS c1, c2 AS c2, c4 AS c3, c5 as c4, c6 AS c5, c7 AS c6, c9 AS c7, c10 AS c8
 into forest8
 from forest10;

COMMIT;
EOF

#MonetDB command
if [ ! -z $MONETDATABASE ]; then
	echo "CREATE TABLE forest10
	(
		c1 double precision, 
		c2 double precision, 
		c3 double precision, 
		c4 double precision, 
		c5 double precision, 
		c6 double precision, 
		c7 double precision, 
		c8 double precision, 
		c9 double precision, 
		c10 double precision
	);
	COPY INTO forest10 FROM '$BASEDIR/raw/covtype.csv' USING DELIMITERS ',';

	CREATE TABLE forest3 AS
      SELECT
         c1 AS c1, c5 AS c2, c9 AS c3
         from forest10
   WITH DATA;
	
   CREATE TABLE forest4 AS
      SELECT
         c2 AS c1, c3 AS c2, c7 AS c3, c9 AS c4
         from forest10
   WITH DATA;
   
   CREATE TABLE forest5 AS
      SELECT
         c2 AS c1, c4 AS c2, c5 AS c3, c6 as c4, c8 AS c5
         from forest10
   WITH DATA;
   
   CREATE TABLE forest8 AS
      SELECT
         c1 AS c1, c2 AS c2, c4 AS c3, c5 as c4, c6 AS c5, c7 AS c6, c9 AS c7, c10 AS c8
         from forest10
   WITH DATA;
   " | mclient -lsql -d$MONETDATABASE 
fi

rm -f $COVTYPE_FILE.tmp
