#/bin/bash
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/../../conf.sh

$PSQL $DATABASE $USER << EOF
	DROP TABLE gen2_d5;
EOF