#/usr/bin/bash
#Creates tpc-h testdata in postgresql
#Needs information about postgresql and the folder containing the dbgen tool 

#/usr/bin/bash
DBGEN_FOLDER=""
PSQL="/usr/local/pgsql/bin/psql"
DATABASE=""
USER=""

#Create tables
cd $DBGEN_FOLDER
./dbgen -f -T S -s 0.167
./dbgen -f -T L -s 0.167

sed 's/|$//' ./partsupp.tbl > ./partsupp_pq.tbl
sed 's/|$//' ./lineitem.tbl > ./lineitem_pq.tbl


#PSQL_COMMAND 
$PSQL $DATABASE $USER << EOF
BEGIN;
CREATE TEMPORARY TABLE PARTSUPP
(
 PS_PARTKEY int,
 PS_SUPPKEY int,
 PS_AVAILQTY double precision,
 PS_SUPPLYCOST double precision,
 PS_COMMENT varchar(199)
);
copy PARTSUPP from '$DBGEN_FOLDER/partsupp_pq.tbl' DELIMITER'|';
ALTER TABLE PARTSUPP ADD PRIMARY KEY (PS_PARTKEY,PS_SUPPKEY);

CREATE TEMPORARY TABLE LINEITEM
(
 L_ORDERKEY int,
 L_PARTKEY int,
 L_SUPPKEY int,
 L_LINENUMBER int,
 L_QUANTITY double precision,
 L_EXTENDEDPRICE double precision,
 L_DISCOUNT double precision,
 L_TAX double precision,
 L_RETURNFLAG char(1),
 L_LINESTATUS char(1),
 L_SHIPDATE date,
 L_COMMITDATE date,
 L_RECEIPTDATE date,
 L_SHIPINSTRUCT char(25),
 L_SHIPMODE char(10),
 L_COMMENT varchar(44)
);
copy LINEITEM from '$DBGEN_FOLDER/lineitem_pq.tbl' DELIMITER'|';
ALTER TABLE LINEITEM ADD FOREIGN KEY (L_PARTKEY,L_SUPPKEY) REFERENCES PARTSUPP(PS_PARTKEY,PS_SUPPKEY);

CREATE TABLE TPCH_DATA
(
 c1 double precision,
 c2 double precision,
 c3 double precision,
 c4 double precision,
 c5 double precision
);
INSERT INTO TPCH_DATA (c1,c2,c3,c4,c5) select L_QUANTITY,L_EXTENDEDPRICE,L_DISCOUNT,L_TAX,PS_AVAILQTY TPCH_DATA FROM PARTSUPP INNER JOIN LINEITEM ON L_PARTKEY = PS_PARTKEY AND L_SUPPKEY = PS_SUPPKEY LIMIT 1000000;
COMMIT;
EOF

rm ./partsupp_pq.tbl
rm ./lineitem_pq.tbl
