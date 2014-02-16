#/usr/bin/bash
#Destroys forest test tables
#Needs information about postgresql 
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

$PSQL $PGDATABASE $USER << EOF
 DROP TABLE FOREST10;
 DROP TABLE FOREST8;
 DROP TABLE FOREST5;
 DROP TABLE FOREST4;
 DROP TABLE FOREST3;
EOF

#MonetDB command
if [ -z $MONETDATABASE ]; then
	exit
fi

echo "
	DROP TABLE FOREST10;
	DROP TABLE FOREST8;
	DROP TABLE FOREST5;
	DROP TABLE FOREST4;
	DROP TABLE FOREST3;" | mclient -lsql -d$MONETDATABASE
