#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# And call the configuration script.
source $DIR/../../conf.sh

# Drop all tables.
echo "DROP TABLE protein10;" > /tmp/load.sql
echo "DROP TABLE protein8;" >> /tmp/load.sql
echo "DROP TABLE protein5;" >> /tmp/load.sql
echo "DROP TABLE protein3;" >> /tmp/load.sql
echo "DROP TABLE protein2;" >> /tmp/load.sql

# Prepare the SQL load script.
echo "CREATE TABLE protein10(" >> /tmp/load.sql
echo "	c1  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c2  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c3  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c4  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c5  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c6  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c7  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c8  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c9  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c10 DOUBLE PRECISION);" >> /tmp/load.sql
echo "COPY protein10 FROM '$DIR/raw/data.csv' DELIMITER'|';" >> /tmp/load.sql

# Prepare the SQL load script for protein8.
echo "SELECT c1 AS c1, " >> /tmp/load.sql
echo "       c2 AS c2," >> /tmp/load.sql
echo "       c3 AS c3," >> /tmp/load.sql
echo "       c5 AS c4," >> /tmp/load.sql
echo "       c6 AS c5," >> /tmp/load.sql
echo "       c7 AS c6," >> /tmp/load.sql
echo "       c9 AS c7," >> /tmp/load.sql
echo "       c10 AS c8 INTO protein8 FROM protein10;" >> /tmp/load.sql

# Prepare the SQL load script for protein5.
echo "SELECT c1 AS c1, " >> /tmp/load.sql
echo "       c3 AS c2," >> /tmp/load.sql
echo "       c4 AS c3," >> /tmp/load.sql
echo "       c8 AS c4," >> /tmp/load.sql
echo "       c9 AS c5 INTO protein5 FROM protein10;" >> /tmp/load.sql

# Prepare the SQL load script for protein3.
echo "SELECT c3 AS c1, " >> /tmp/load.sql
echo "       c4 AS c2," >> /tmp/load.sql
echo "       c5 AS c3 INTO protein3 FROM protein10;" >> /tmp/load.sql

# Prepare the SQL load script for protein2.
echo "SELECT c2 AS c1, " >> /tmp/load.sql
echo "       c7 AS c2 INTO protein2 FROM protein10;" >> /tmp/load.sql


# Now call the load script.
$PSQL $PGDATABASE $USER -f /tmp/load.sql
