#/bin/bash

# Postgres specific variables:
PGPORT=""
PGDATAFOLDER=""
PGDATABASE=""

if [ -z "$PGDATAFOLDER" ]; then
  echo 'Please provide a $PGDATAFOLDER in conf.sh.'
  exit
fi

# Set default values.
if [ -z "$PGPORT" ]; then
  PGPORT="5432"
fi
if [ -z "$PGDATABASE" ]; then
  PGDATABASE=`whoami`
fi
