#/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../conf.sh

# If MonetDB is used, make sure that the database is read-only (drastically improves query performance).
if [ ! -z $MONET_DATABASE ]; then
	monetdb stop $MONET_DATABASE
	monetdb set readonly=true $MONET_DATABASE
fi

forest/create-workload.sh
set1/create-workload.sh
set2/create-workload.sh
set3/create-workload.sh
set4/create-workload.sh
set5/create-workload.sh

