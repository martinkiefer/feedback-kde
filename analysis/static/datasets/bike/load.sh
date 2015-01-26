#!/bin/bash

# Figure out the current directory.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# And call the configuration script.
source $DIR/../../conf.sh

# Drop all tables.
echo "DROP TABLE bike16;" > /tmp/load.sql
echo "DROP TABLE bike8;" >> /tmp/load.sql
echo "DROP TABLE bike5;" >> /tmp/load.sql
echo "DROP TABLE bike3;" >> /tmp/load.sql
echo "DROP TABLE bike2;" >> /tmp/load.sql

# Prepare the SQL load script for bike16.
echo "CREATE TABLE bike16(" >> /tmp/load.sql
echo "	c1  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c2  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c3  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c4  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c5  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c6  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c7  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c8  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c9  DOUBLE PRECISION," >> /tmp/load.sql
echo "	c10 DOUBLE PRECISION," >> /tmp/load.sql
echo "	c11 DOUBLE PRECISION," >> /tmp/load.sql
echo "	c12 DOUBLE PRECISION," >> /tmp/load.sql
echo "	c13 DOUBLE PRECISION," >> /tmp/load.sql
echo "	c14 DOUBLE PRECISION," >> /tmp/load.sql
echo "	c15 DOUBLE PRECISION," >> /tmp/load.sql
echo "	c16 DOUBLE PRECISION);" >> /tmp/load.sql
echo "COPY bike16 FROM '$DIR/raw/data.csv' DELIMITER'|';" >> /tmp/load.sql

# Prepare the SQL load script for bike8.
echo "SELECT c1 AS c1, " >> /tmp/load.sql
echo "       c5 AS c2," >> /tmp/load.sql
echo "       c6 AS c3," >> /tmp/load.sql
echo "       c8 AS c4," >> /tmp/load.sql
echo "       c9 AS c5," >> /tmp/load.sql
echo "       c10 AS c6," >> /tmp/load.sql
echo "       c11 AS c7," >> /tmp/load.sql
echo "       c12 AS c8 INTO bike8 FROM bike16;" >> /tmp/load.sql

# Prepare the SQL load script for bike5.
echo "SELECT c1 AS c1, " >> /tmp/load.sql
echo "       c9 AS c2," >> /tmp/load.sql
echo "       c10 AS c3," >> /tmp/load.sql
echo "       c11 AS c4," >> /tmp/load.sql
echo "       c12 AS c5 INTO bike5 FROM bike16;" >> /tmp/load.sql

# Prepare the SQL load script for bike3.
echo "SELECT c1 AS c1, " >> /tmp/load.sql
echo "       c5 AS c2," >> /tmp/load.sql
echo "       c6 AS c3 INTO bike3 FROm bike16;" >> /tmp/load.sql

# Prepare the SQL load script for bike2.
echo "SELECT c10 AS c1, " >> /tmp/load.sql
echo "       c12 AS c2 INTO bike2 FROm bike16;" >> /tmp/load.sql

# Now call the load script.
psql -p$PGPORT $PGDATABASE -f /tmp/load.sql
