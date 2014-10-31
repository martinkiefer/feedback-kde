#/bin/bash

# First, run the forest workload.
DIMENSIONS=(8)
SAMPLESIZES=(1024)
PARAMETERS=(0)
QUERIES=1000
REPETITIONS=0
ERROR="absolute"
MAINTENANCE="none"

PG_DATA=/home/mheimel/postgres/data

# First, run the workloads without sample tracking. 
#REPETITIONS=4
#OPTIMIZATION=(heuristic stholes adaptive)
OPTIMIZATION=(stholes adaptive)
for i in `seq 0 $REPETITIONS`
do
   for samplesize in "${SAMPLESIZES[@]}"
	do
      for dimension in "${DIMENSIONS[@]}"
		do
         for optimization in "${OPTIMIZATION[@]}"
         do
			   postgres -D $PG_DATA 2> postgres.err > postgres.out &
            PID=$!
            sleep 5
            bash ./mvtc_id/load-mvtc_id-tables.sh
				python runExperiment.py	--dbname=mheimel --dataset=mvtc_id \
               --dimensions=$dimension --samplesize=$samplesize \
					--error=$ERROR --optimization=$optimization \
					--log=$dimension"_"$optimization".log" \
               --sample_maintenance=$MAINTENANCE
            kill -9 $PID
            sleep 5
            cp /tmp/error.log $optimization.log
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
			postgres -D $PG_DATA 2> postgres.err > postgres.out &
         PID=$!
         sleep 5
		   bash ./mvtc_id/load-mvtc_id-tables.sh
			python runExperiment.py	--dbname=mheimel --dataset=mvtc_id \
            --dimensions=$dimension --samplesize=$samplesize \
				--error=$ERROR --optimization=adaptive \
				--log=$dimension"_adaptive_periodic.log" \
            --sample_maintenance=periodic --period=2
         kill -9 $PID
        sleep 5
         cp /tmp/error.log adaptive_periodic.log
		done
	done
done
