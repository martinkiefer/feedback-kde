#/usr/bin/bash
#Destroys forest test tables
#Needs information about postgresql 

PSQL="/usr/local/pgsql/bin/psql"
DATABASE=""
USER=""

$PSQL $DATABASE $USER << EOF
 DROP TABLE FOREST10;
 DROP TABLE FOREST8;
 DROP TABLE FOREST5;
 DROP TABLE FOREST4;
 DROP TABLE FOREST3;
EOF
