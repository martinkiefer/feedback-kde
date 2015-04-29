#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# And call the configuration script.
source $DIR/../../../conf.sh

# Drop all tables.
echo "DROP TABLE IF EXISTS power9;" > /tmp/load.sql
echo "DROP TABLE IF EXISTS power8;" >> /tmp/load.sql
echo "DROP TABLE IF EXISTS power5;" >> /tmp/load.sql
echo "DROP TABLE IF EXISTS power3;" >> /tmp/load.sql
echo "DROP TABLE IF EXISTS power2;" >> /tmp/load.sql

# Prepare the SQL load script.
echo "CREATE TABLE power9(" >> /tmp/load.sql
echo "	c1  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c2  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c3  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c4  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c5  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c6  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c7  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c8  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c9  DOUBLE PRECISION);" >> /tmp/load.sql
echo "COPY power9 FROM '$DIR/raw/data.csv' DELIMITER'|';" >> /tmp/load.sql

# Prepare the SQL load script for power8.
echo "SELECT c1 AS c1, " >> /tmp/load.sql
echo "       c2 AS c2," >> /tmp/load.sql
echo "       c3 AS c3," >> /tmp/load.sql
echo "       c4 AS c4," >> /tmp/load.sql
echo "       c5 AS c5," >> /tmp/load.sql
echo "       c6 AS c6," >> /tmp/load.sql
echo "       c7 AS c7," >> /tmp/load.sql
echo "       c8 AS c8 INTO power8 FROM power9;" >> /tmp/load.sql

# Prepare the SQL load script for power5.
echo "SELECT c1 AS c1, " >> /tmp/load.sql
echo "       c3 AS c2," >> /tmp/load.sql
echo "       c4 AS c3," >> /tmp/load.sql
echo "       c6 AS c4," >> /tmp/load.sql
echo "       c9 AS c5 INTO power5 FROM power9;" >> /tmp/load.sql

# Prepare the SQL load script for power3.
echo "SELECT c3 AS c1, " >> /tmp/load.sql
echo "       c4 AS c2," >> /tmp/load.sql
echo "       c5 AS c3 INTO power3 FROM power9;" >> /tmp/load.sql

# Prepare the SQL load script for power2.
echo "SELECT c2 AS c1, " >> /tmp/load.sql
echo "       c5 AS c2 INTO power2 FROM power9;" >> /tmp/load.sql

# Now call the load script.
$PSQL -p$PGPORT $PGDATABASE -f /tmp/load.sql 
