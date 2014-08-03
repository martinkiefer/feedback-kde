#!/usr/bin/bash

#Sample code showing how VisualizeSampleMaintenace.py can be applied on an existing estimator. 

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $BASEDIR/conf.sh

#Load the tables
bash $BASEDIR/genhist/set1/load-set1-tables.sh

#Build the estimator
$PSQL $PGDATABASE $USER << EOF
  SET kde_samplesize TO 512;
  SET kde_error_metric TO Quadratic;
  SET ocl_use_gpu TO false;
  SET kde_debug TO false;
  SET kde_enable TO true;
  ANALYZE gen1_d2(c1,c2);
EOF

python2 $BASEDIR/VisualizeSampleMaintenace.py --dbname xy --query_file dv.sql --table gen1_d2 --resolution 5000 --steps 5 \
      --queries_per_step 25 --folder /tmp --sample_maintenance none --delete_constraint "c1 > 0.5 and c2 > 0.5" 

