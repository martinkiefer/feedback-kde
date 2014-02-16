#/usr/bin/bash
#Destroys tpch test tables
#Needs information about postgresql 
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

$PSQL $PGDATABASE $USER << EOF
 DROP TABLE TPCH_DATA;
EOF

#MonetDB command
if [ -z $MONETDATABASE ]; then
	exit
fi

echo "DROP TABLE TPCH_DATA; DROP TABLE LINEITEM; DROP TABLE PARTSUPP;" | mclient -lsql -d$MONETDATABASE