#/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

# If MonetDB is used, make sure that the database is writable.
if [ ! -z $MONET_DATABASE ]; then
	monetdb stop $MONET_DATABASE
	monetdb set readonly=false $MONET_DATABASE
fi

forest/load-forest-tables.sh
set1/load-set1-tables.sh
set2/load-set2-tables.sh
set3/load-tpch-tables.sh