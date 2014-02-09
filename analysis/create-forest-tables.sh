#/usr/bin/bash
#Creates forest cover testdata in postgresql
#Needs information about postgresql and the file containing the forest cover dataset

#/usr/bin/bash
COVTYPE_FILE=""
PSQL="/usr/local/pgsql/bin/psql"
DATABASE=""
USER=""

#Extract first 10 columns
cat $COVTYPE_FILE | cut -d , -f 1-10 > $COVTYPE_FILE.tmp

#Put in temporary table and create permanent tables with noramlized values
$PSQL $DATABASE $USER << EOF
BEGIN;
CREATE TEMPORARY TABLE T
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
copy T from '$COVTYPE_FILE.tmp' DELIMITER',' CSV;

SELECT
 (c1-c1min)/(c1max-c1min) as c1,
 (c2-c2min)/(c2max-c2min) as c2,
 (c3-c3min)/(c3max-c3min) as c3
 into forest3
 from T,
 (
   SELECT
   (SELECT min(c1) FROM T) AS c1min,
   (SELECT max(c1) FROM T) AS c1max,
   (SELECT min(c2) FROM T) AS c2min,
   (SELECT max(c2) FROM T) AS c2max,
   (SELECT min(c3) FROM T) AS c3min,
   (SELECT max(c3) FROM T) AS c3max
 ) as minmax;

SELECT
 (c1-c1min)/(c1max-c1min) as c1,
 (c2-c2min)/(c2max-c2min) as c2,
 (c3-c3min)/(c3max-c3min) as c3,
 (c4-c4min)/(c4max-c4min) as c4
 into forest4
 from T,
 (
   SELECT
   (SELECT min(c1) FROM T) AS c1min,
   (SELECT max(c1) FROM T) AS c1max,
   (SELECT min(c2) FROM T) AS c2min,
   (SELECT max(c2) FROM T) AS c2max,
   (SELECT min(c3) FROM T) AS c3min,
   (SELECT max(c3) FROM T) AS c3max,
   (SELECT min(c4) FROM T) AS c4min,
   (SELECT max(c4) FROM T) AS c4max
 ) as minmax;

SELECT
 (c1-c1min)/(c1max-c1min) as c1,
 (c2-c2min)/(c2max-c2min) as c2,
 (c3-c3min)/(c3max-c3min) as c3,
 (c4-c4min)/(c4max-c4min) as c4,
 (c5-c5min)/(c5max-c5min) as c5
 into forest5
 from T,
 (
   SELECT
   (SELECT min(c1) FROM T) AS c1min,
   (SELECT max(c1) FROM T) AS c1max,
   (SELECT min(c2) FROM T) AS c2min,
   (SELECT max(c2) FROM T) AS c2max,
   (SELECT min(c3) FROM T) AS c3min,
   (SELECT max(c3) FROM T) AS c3max,
   (SELECT min(c4) FROM T) AS c4min,
   (SELECT max(c4) FROM T) AS c4max,
   (SELECT min(c5) FROM T) AS c5min,
   (SELECT max(c5) FROM T) AS c5max
 ) as minmax;

SELECT
 (c1-c1min)/(c1max-c1min) as c1,
 (c2-c2min)/(c2max-c2min) as c2,
 (c3-c3min)/(c3max-c3min) as c3,
 (c4-c4min)/(c4max-c4min) as c4,
 (c5-c5min)/(c5max-c5min) as c5,
 (c6-c6min)/(c6max-c6min) as c6,
 (c7-c7min)/(c7max-c7min) as c7,
 (c8-c8min)/(c8max-c8min) as c8
 into forest8
 from T,
 (
   SELECT
   (SELECT min(c1) FROM T) AS c1min,
   (SELECT max(c1) FROM T) AS c1max,
   (SELECT min(c2) FROM T) AS c2min,
   (SELECT max(c2) FROM T) AS c2max,
   (SELECT min(c3) FROM T) AS c3min,
   (SELECT max(c3) FROM T) AS c3max,
   (SELECT min(c4) FROM T) AS c4min,
   (SELECT max(c4) FROM T) AS c4max,
   (SELECT min(c5) FROM T) AS c5min,
   (SELECT max(c5) FROM T) AS c5max,
   (SELECT min(c6) FROM T) AS c6min,
   (SELECT max(c6) FROM T) AS c6max,
   (SELECT min(c7) FROM T) AS c7min,
   (SELECT max(c7) FROM T) AS c7max,
   (SELECT min(c8) FROM T) AS c8min,
   (SELECT max(c8) FROM T) AS c8max
 ) as minmax;


SELECT
 (c1-c1min)/(c1max-c1min)*100 as c1,
 (c2-c2min)/(c2max-c2min)*100 as c2,
 (c3-c3min)/(c3max-c3min)*100 as c3,
 (c4-c4min)/(c4max-c4min)*100 as c4,
 (c5-c5min)/(c5max-c5min)*100 as c5,
 (c6-c6min)/(c6max-c6min)*100 as c6,
 (c7-c7min)/(c7max-c7min)*100 as c7,
 (c8-c8min)/(c8max-c8min)*100 as c8,
 (c9-c9min)/(c9max-c9min)*100 as c9,
 (c10-c10min)/(c10max-c10min)*100 as c10
 into forest10
 from T,
 (
   SELECT
   (SELECT min(c1) FROM T) AS c1min,
   (SELECT max(c1) FROM T) AS c1max,
   (SELECT min(c2) FROM T) AS c2min,
   (SELECT max(c2) FROM T) AS c2max,
   (SELECT min(c3) FROM T) AS c3min,
   (SELECT max(c3) FROM T) AS c3max,
   (SELECT min(c4) FROM T) AS c4min,
   (SELECT max(c4) FROM T) AS c4max,
   (SELECT min(c5) FROM T) AS c5min,
   (SELECT max(c5) FROM T) AS c5max,
   (SELECT min(c6) FROM T) AS c6min,
   (SELECT max(c6) FROM T) AS c6max,
   (SELECT min(c7) FROM T) AS c7min,
   (SELECT max(c7) FROM T) AS c7max,
   (SELECT min(c8) FROM T) AS c8min,
   (SELECT max(c8) FROM T) AS c8max,
   (SELECT min(c9) FROM T) AS c9min,
   (SELECT max(c9) FROM T) AS c9max,
   (SELECT min(c10) FROM T) AS c10min,
   (SELECT max(c10) FROM T) AS c10max
 ) as minmax;
COMMIT;
EOF

rm -f $COVTYPE_FILE.tmp
