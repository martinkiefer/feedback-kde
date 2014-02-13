#/usr/bin/bash
#Destroys tpch test tables
#Needs information about postgresql 

PSQL="/usr/local/pgsql/bin/psql"
DATABASE=""
USER=""

$PSQL $DATABASE $USER << EOF
 DROP TABLE TPCH_DATA;
EOF
