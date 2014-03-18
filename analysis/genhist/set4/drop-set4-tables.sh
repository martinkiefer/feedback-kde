#/bin/bash
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

$PSQL $PGDATABASE $USER << EOF
	DROP TABLE gen4_d10;
	DROP TABLE gen4_d8;
	DROP TABLE gen4_d5;
	DROP TABLE gen4_d4;
	DROP TABLE gen4_d3;
EOF

#MonetDB command
if [ -z $MONETDATABASE ]; then
	exit
fi

echo "
	DROP TABLE gen4_d10;
	DROP TABLE gen4_d8;
	DROP TABLE gen4_d5;
	DROP TABLE gen4_d4;
	DROP TABLE gen4_d3;" | mclient -lsql -d$MONETDATABASE