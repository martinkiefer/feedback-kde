#/bin/bash
PGDATA=/home/mkiefer/current_data

# First, run the forest workload.
DATASETS=("mvtc_id")
DIMENSIONS=(5)
WORKLOADS=(2)
SAMPLESIZES=(2048)
PARAMETERS=(0)
QUERIES=1000
REPETITIONS=0
ERROR="absolute"
MAINTENANCE="none"


REPETITIONS=-1
for i in `seq 0 $REPETITIONS`
do
	for dataset in "${DATASETS[@]}"
	do
		for workload in "${WORKLOADS[@]}"
		do
			for samplesize in "${SAMPLESIZES[@]}"
			do
				for dimension in "${DIMENSIONS[@]}"
				do
					for par in "${PARAMETERS[@]}"
				do                                        
					#/home/mkiefer/postgres/bin/postgres -D $PGDATA 2>> postgres.log &
                                        #PID=$!
                                        #sleep 3
					bash ./mvtc_id/load-mvtc_id-tables.sh
					python2.6 runExperiment.py	--dbname=xy --dataset=$dataset --dimensions=$dimension \
										--samplesize=$samplesize \
										--error=$ERROR --optimization=none \
											--log=$dimension"_"$samplesize"_"$MAINTENANCE"_"$par --sample_maintenance=$MAINTENANCE
                                        #kill -9 $PID
                                        #sleep 1
					done
				done
			done
		done
	done
done


REPETITIONS=0
MAINTENANCE="threshold"
PARAMETERS=(0.00025 0.0005 0.001)
for i in `seq 0 $REPETITIONS`
do
	for dataset in "${DATASETS[@]}"
	do
		for workload in "${WORKLOADS[@]}"
		do
			for samplesize in "${SAMPLESIZES[@]}"
			do
				for dimension in "${DIMENSIONS[@]}"
				do
					for par in "${PARAMETERS[@]}"
					do
                                        #/home/mkiefer/postgres/bin/postgres -D $PGDATA 2>> postgres.log &
                                        #PID=$!
                                        #sleep 3
					bash ./mvtc_id/load-mvtc_id-tables.sh
					python2.6 runExperiment.py	--dbname=xy --dataset=$dataset --dimensions=$dimension \
											--samplesize=$samplesize \
											--error=$ERROR --optimization=none --threshold=$par \
											--log=$dimension"_"$samplesize"_"$MAINTENANCE"_"$par --sample_maintenance=$MAINTENANCE
                                        #kill -9 $PID
                                        #sleep 1
					done
				done
			done
		done
	done
done

REPETITIONS=0
MAINTENANCE="periodic"
PARAMETERS=(5 10 20)
for i in `seq 0 $REPETITIONS`
do
	for dataset in "${DATASETS[@]}"
	do
		for workload in "${WORKLOADS[@]}"
		do
			for samplesize in "${SAMPLESIZES[@]}"
			do
				for dimension in "${DIMENSIONS[@]}"
				do
					for par in "${PARAMETERS[@]}"
					do
                                        #/home/mkiefer/postgres/bin/postgres -D $PGDATA 2>> postgres.log &
                                        #PID=$!
                                        #sleep 3
					bash ./mvtc_id/load-mvtc_id-tables.sh
					python2.6 runExperiment.py	--dbname=xy --dataset=$dataset --dimensions=$dimension \
											--samplesize=$samplesize \
											--error=$ERROR --optimization=none --period=$par\
											--log=$dimension"_"$samplesize"_"$MAINTENANCE"_"$par --sample_maintenance=$MAINTENANCE
                                        #kill -9 $PID
                                        #sleep 1
					done
				done
			done
		done
	done
done
