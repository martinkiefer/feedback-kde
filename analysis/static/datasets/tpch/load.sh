#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# And call the configuration script.
source $DIR/../../../conf.sh

# Prepare the required tpch tables.
$PSQL -p$PGPORT $PGDATABASE -f $DIR/dbgen/tpch_dll.sql
echo "COPY lineitem FROM '$DIR/raw/lineitem.tbl' DELIMITER '|' CSV;" > /tmp/load.sql
echo "COPY partsupp FROM '$DIR/raw/partsupp.tbl' DELIMITER '|' CSV;" >> /tmp/load.sql
$PSQL -p$PGPORT $PGDATABASE -f /tmp/load.sql

# Create the joined table.
echo "DROP TABLE IF EXISTS tpch;" > /tmp/load.sql
echo "CREATE TABLE tpch (c1 FLOAT, c2 FLOAT, c3 FLOAT, c4 FLOAT, c5 FLOAT);" >> /tmp/load.sql
echo "INSERT INTO tpch SELECT l_quantity, l_extendedprice, l_discount, l_tax, ps_availqty FROM lineitem JOIN partsupp ON (l_partkey = ps_partkey AND l_suppkey = ps_suppkey);" >> /tmp/load.sql
$PSQL -p$PGPORT $PGDATABASE -f /tmp/load.sql

# Clean up.
echo "DROP TABLE IF EXISTS lineitem;" > /tmp/cleanup.sql 
echo "DROP TABLE IF EXISTS part;" >> /tmp/cleanup.sql 
echo "DROP TABLE IF EXISTS partsupp;" >> /tmp/cleanup.sql 
echo "DROP TABLE IF EXISTS region;" >> /tmp/cleanup.sql 
echo "DROP TABLE IF EXISTS nation;" >> /tmp/cleanup.sql 
echo "DROP TABLE IF EXISTS orders;" >> /tmp/cleanup.sql 
echo "DROP TABLE IF EXISTS customer;" >> /tmp/cleanup.sql 
echo "DROP TABLE IF EXISTS supplier;" >> /tmp/cleanup.sql 
$PSQL -p$PGPORT $PGDATABASE -f /tmp/cleanup.sql

