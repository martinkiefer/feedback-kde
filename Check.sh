#!/usr/bin/bash
source ./analysis/conf.sh

rm -f logfile
PGBIN=$PGINSTFOLDER/bin
$PGBIN/pg_ctl -D $PGDATAFOLDER -l logfile start
sleep 3
$PYTHON Check.py --dbname $PGDATABASE --port $PGPORT
$PGBIN/pg_ctl -D $PGDATAFOLDER -l logfile stop
