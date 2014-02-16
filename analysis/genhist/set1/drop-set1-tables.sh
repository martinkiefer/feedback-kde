#/bin/bash
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

$PSQL $PGDATABASE $USER << EOF
	DROP TABLE gen1_d10;
	DROP TABLE gen1_d8;
	DROP TABLE gen1_d5;
	DROP TABLE gen1_d4;
	DROP TABLE gen1_d3;
EOF

#MonetDB command
if [ -z $MONETDATABASE ]; then
	exit
fi

echo "
	DROP TABLE gen1_d10;
	DROP TABLE gen1_d8;
	DROP TABLE gen1_d5;
	DROP TABLE gen1_d4;
	DROP TABLE gen1_d3;" | mclient -lsql -d$MONETDATABASE