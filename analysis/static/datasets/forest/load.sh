#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# And call the configuration script.
source $DIR/../../../conf.sh

# Drop all tables.
echo "DROP TABLE IF EXISTS forest10;" > /tmp/load.sql
echo "DROP TABLE IF EXISTS forest8;" >> /tmp/load.sql
echo "DROP TABLE IF EXISTS forest5;" >> /tmp/load.sql
echo "DROP TABLE IF EXISTS forest3;" >> /tmp/load.sql
echo "DROP TABLE IF EXISTS forest2;" >> /tmp/load.sql

# Prepare the SQL load script.
echo "CREATE TABLE forest10(" >> /tmp/load.sql
echo "	c1  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c2  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c3  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c4  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c5  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c6  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c7  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c8  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c9  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c10  DOUBLE PRECISION);" >> /tmp/load.sql
echo "COPY forest10 FROM '$DIR/raw/data.csv' DELIMITER'|';" >> /tmp/load.sql

# Prepare the SQL load script for forest8.
echo "SELECT c1 AS c1, " >> /tmp/load.sql
echo "       c2 AS c2," >> /tmp/load.sql
echo "       c4 AS c3," >> /tmp/load.sql
echo "       c5 AS c4," >> /tmp/load.sql
echo "       c6 AS c5," >> /tmp/load.sql
echo "       c7 AS c6," >> /tmp/load.sql
echo "       c9 AS c7," >> /tmp/load.sql
echo "       c10 AS c8 INTO forest8 FROM forest10;" >> /tmp/load.sql

# Prepare the SQL load script for forest5.
echo "SELECT c2 AS c1, " >> /tmp/load.sql
echo "       c4 AS c2," >> /tmp/load.sql
echo "       c5 AS c3," >> /tmp/load.sql
echo "       c6 AS c4," >> /tmp/load.sql
echo "       c8 AS c5 INTO forest5 FROM forest10;" >> /tmp/load.sql

# Prepare the SQL load script for forest3.
echo "SELECT c1 AS c1, " >> /tmp/load.sql
echo "       c5 AS c2," >> /tmp/load.sql
echo "       c9 AS c3 INTO forest3 FROM forest10;" >> /tmp/load.sql

# Prepare the SQL load script for forest2.
echo "SELECT c2 AS c1, " >> /tmp/load.sql
echo "       c6 AS c2 INTO forest2 FROM forest10;" >> /tmp/load.sql

# Now call the load script.
$PSQL -p$PGPORT $PGDATABASE -f /tmp/load.sql 
