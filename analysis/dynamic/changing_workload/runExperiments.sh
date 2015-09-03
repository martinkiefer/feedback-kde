#/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/../../conf.sh

# First, run the forest workload.
DIMENSIONS=(5 8)
SAMPLESIZES=(1024)
ERROR="absolute"
MAINTENANCE="none"
LOGPREFIX=$DIR/../../evaluation/changing_data/

# First, run the workloads without sample tracking. 
REPETITIONS=0
OPTIMIZATION=(heuristic stholes)

for i in `seq 0 $REPETITIONS`
do
   for samplesize in "${SAMPLESIZES[@]}"
	do
      for dimension in "${DIMENSIONS[@]}"
		do
         for optimization in "${OPTIMIZATION[@]}"
         do
			   $POSTGRES -D $PGDATAFOLDER -p $PGPORT 2>> postgres.err >> postgres.out &
            PID=$!
            sleep 5
            bash ./mvtc_id/load-mvtc_id-tables.sh
				$PYTHON runExperiment.py	--dbname=$PGDATABASE --port=$PGPORT --dataset=mvtc_id \
               --dimensions=$dimension --samplesize=$samplesize \
					--error=$ERROR --optimization=$optimization \
					--log=$dimension"_"$optimization".log" \
               --sample_maintenance=$MAINTENANCE
            kill -9 $PID
            sleep 5
            cp /tmp/error.log  $LOGPREFIX$dimension"_"$optimization"_raw.log"
			done
		done
	done
done

for i in `seq 0 $REPETITIONS`
do
   for samplesize in "${SAMPLESIZES[@]}"
	do
	   for dimension in "${DIMENSIONS[@]}"
		do
			$POSTGRES -D $PGDATAFOLDER -p $PGPORT 2>> postgres.err >> postgres.out &
         PID=$!
         sleep 5
		   bash ./mvtc_id/load-mvtc_id-tables.sh
			$PYTHON runExperiment.py	--dbname=$PGDATABASE --port=$PGPORT  --dataset=mvtc_id \
            --dimensions=$dimension --samplesize=$samplesize \
				--error=$ERROR --optimization=adaptive \
				--log=$dimension"_adaptive_periodic.log" \
            --sample_maintenance=tkr --threshold -0.5
         kill -9 $PID
        sleep 5
         cp /tmp/error.log $LOGPREFIX$dimension"_adaptive_periodic_raw.log"
		done
	done
done
