#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# And call the configuration script.
source $DIR/../../conf.sh

# Drop all tables.
echo "DROP TABLE set1_8;" > /tmp/load.sql
echo "DROP TABLE set1_5;" >> /tmp/load.sql
echo "DROP TABLE set1_3;" >> /tmp/load.sql
echo "DROP TABLE set1_2;" >> /tmp/load.sql

# Prepare the SQL load script for set1_8.
echo "CREATE TABLE set1_8(" >> /tmp/load.sql
echo "	c1  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c2  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c3  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c4  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c5  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c6  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c7  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c8  DOUBLE PRECISION);" >> /tmp/load.sql
echo "COPY set1_8 FROM '$DIR/raw/gen1_d8.csv' DELIMITER'|';" >> /tmp/load.sql

# Prepare the SQL load script for set1_5.
echo "CREATE TABLE set1_5(" >> /tmp/load.sql
echo "	c1  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c2  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c3  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c4  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c5  DOUBLE PRECISION);" >> /tmp/load.sql
echo "COPY set1_5 FROM '$DIR/raw/gen1_d5.csv' DELIMITER'|';" >> /tmp/load.sql

# Prepare the SQL load script for set1_3.
echo "CREATE TABLE set1_3(" >> /tmp/load.sql
echo "	c1  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c2  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c3  DOUBLE PRECISION);" >> /tmp/load.sql
echo "COPY set1_3 FROM '$DIR/raw/gen1_d3.csv' DELIMITER'|';" >> /tmp/load.sql

# Prepare the SQL load script for set1_2.
echo "CREATE TABLE set1_2(" >> /tmp/load.sql
echo "	c1  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c2  DOUBLE PRECISION);" >> /tmp/load.sql
echo "COPY set1_2 FROM '$DIR/raw/gen1_d2.csv' DELIMITER'|';" >> /tmp/load.sql

# Now call the load script.
$PSQL $PGDATABASE $USER -f /tmp/load.sql
