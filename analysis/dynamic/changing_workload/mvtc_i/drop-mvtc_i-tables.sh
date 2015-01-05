#/bin/bash
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

$PSQL $PGDATABASE $USER << EOF
	DROP TABLE mvtc_i_d10;
	DROP TABLE mvtc_i_d8;
	DROP TABLE mvtc_i_d5;
	DROP TABLE mvtc_i_d4;
	DROP TABLE mvtc_i_d3;
EOF

#MonetDB command
if [ -z $MONETDATABASE ]; then
	exit
fi

echo "
	DROP TABLE mvtc_i_d10;
	DROP TABLE mvtc_i_d8;
	DROP TABLE mvtc_i_d5;
	DROP TABLE mvtc_i_d4;
	DROP TABLE mvtc_i_d3;" | mclient -lsql -d$MONETDATABASE
