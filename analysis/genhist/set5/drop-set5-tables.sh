#/bin/bash
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

$PSQL $PGDATABASE $USER << EOF
	DROP TABLE gen5_d5;
EOF

#MonetDB command
if [ -z $MONETDATABASE ]; then
	exit
fi

echo "DROP TABLE gen5_d5;" | mclient -lsql -d$MONETDATABASE