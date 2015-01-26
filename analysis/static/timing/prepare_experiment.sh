#!/bin/bash

# Figure out the current directory.                                             
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"                         
# And call the configuration script.                                            
source $DIR/../../conf.sh

# Prepare a query file.
QUERYFILE="/tmp/load.$$.tmp"

# Drop all tables.
echo "DROP TABLE time2;" > $QUERYFILE
echo "DROP TABLE time3;" >> $QUERYFILE
echo "DROP TABLE time4;" >> $QUERYFILE
echo "DROP TABLE time5;" >> $QUERYFILE
echo "DROP TABLE time6;" >> $QUERYFILE
echo "DROP TABLE time7;" >> $QUERYFILE
echo "DROP TABLE time8;" >> $QUERYFILE

# Create the tables.
echo "CREATE TABLE time8(" >> $QUERYFILE
echo "   c1  DOUBLE PRECISION," >> $QUERYFILE
echo "   c2  DOUBLE PRECISION," >> $QUERYFILE
echo "   c3  DOUBLE PRECISION," >> $QUERYFILE
echo "   c4  DOUBLE PRECISION," >> $QUERYFILE
echo "   c5  DOUBLE PRECISION," >> $QUERYFILE
echo "   c6  DOUBLE PRECISION," >> $QUERYFILE
echo "   c7  DOUBLE PRECISION," >> $QUERYFILE
echo "   c8  DOUBLE PRECISION);" >> $QUERYFILE
echo "CREATE TABLE time7(" >> $QUERYFILE
echo "   c1  DOUBLE PRECISION," >> $QUERYFILE
echo "   c2  DOUBLE PRECISION," >> $QUERYFILE
echo "   c3  DOUBLE PRECISION," >> $QUERYFILE
echo "   c4  DOUBLE PRECISION," >> $QUERYFILE
echo "   c5  DOUBLE PRECISION," >> $QUERYFILE
echo "   c6  DOUBLE PRECISION," >> $QUERYFILE
echo "   c7  DOUBLE PRECISION);" >> $QUERYFILE
echo "CREATE TABLE time6(" >> $QUERYFILE
echo "   c1  DOUBLE PRECISION," >> $QUERYFILE
echo "   c2  DOUBLE PRECISION," >> $QUERYFILE
echo "   c3  DOUBLE PRECISION," >> $QUERYFILE
echo "   c4  DOUBLE PRECISION," >> $QUERYFILE
echo "   c5  DOUBLE PRECISION," >> $QUERYFILE
echo "   c6  DOUBLE PRECISION);" >> $QUERYFILE
echo "CREATE TABLE time5(" >> $QUERYFILE
echo "   c1  DOUBLE PRECISION," >> $QUERYFILE
echo "   c2  DOUBLE PRECISION," >> $QUERYFILE
echo "   c3  DOUBLE PRECISION," >> $QUERYFILE
echo "   c4  DOUBLE PRECISION," >> $QUERYFILE
echo "   c5  DOUBLE PRECISION);" >> $QUERYFILE
echo "CREATE TABLE time4(" >> $QUERYFILE
echo "   c1  DOUBLE PRECISION," >> $QUERYFILE
echo "   c2  DOUBLE PRECISION," >> $QUERYFILE
echo "   c3  DOUBLE PRECISION," >> $QUERYFILE
echo "   c4  DOUBLE PRECISION);" >> $QUERYFILE
echo "CREATE TABLE time3(" >> $QUERYFILE
echo "   c1  DOUBLE PRECISION," >> $QUERYFILE
echo "   c2  DOUBLE PRECISION," >> $QUERYFILE
echo "   c3  DOUBLE PRECISION);" >> $QUERYFILE
echo "CREATE TABLE time2(" >> $QUERYFILE
echo "   c1  DOUBLE PRECISION," >> $QUERYFILE
echo "   c2  DOUBLE PRECISION);" >> $QUERYFILE

# Now fill the tables with ~2.100.000 rows each.
echo "INSERT INTO time8 SELECT * FROM powernormalized8;" >> $QUERYFILE
echo "INSERT INTO time8 SELECT * FROM forestnormalized8 LIMIT 100000;" >> $QUERYFILE
echo "INSERT INTO time7 SELECT c1,c2,c3,c4,c5,c6,c7 FROM time8;" >> $QUERYFILE
echo "INSERT INTO time6 SELECT c1,c2,c3,c4,c5,c6 FROM time8;" >> $QUERYFILE
echo "INSERT INTO time5 SELECT c1,c2,c3,c4,c5 FROM time8;" >> $QUERYFILE
echo "INSERT INTO time4 SELECT c1,c2,c3,c4 FROM time8;" >> $QUERYFILE
echo "INSERT INTO time3 SELECT c1,c2,c3 FROM time8;" >> $QUERYFILE
echo "INSERT INTO time2 SELECT c1,c2 FROM time8;" >> $QUERYFILE

# Finally, call the load script.
psql -p$PGPORT $PGDATABASE -f $QUERYFILE
rm $QUERYFILE
