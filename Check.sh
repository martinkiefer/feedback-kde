#!/usr/bin/bash
source ./analysis/conf.sh

PGBIN=$PGINSTFOLDE/bin
$PGBIN/pg_ctl -D $PGDATAFOLDER -l logfile start
sleep 3
$PYTHON --dbname $PGDATABASE --port $PGPORT
$PGBIN/pg_ctl -D $PGDATAFOLDER -l logfile stop
