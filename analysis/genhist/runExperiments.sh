#/bin/bash

LOG="/home/mheimel/experiments_vldb.log"

# First, run the forest workload.
DATASETS=("set1")
DIM="5"
WORKLOADS=(3)
SAMPLESIZES=(2048 4096 8192 16384 32768)
BATCHSIZES=(5)
VARIANTS=("adaptive" "batch_workload")
QUERIES=1500
REPETITIONS=4
TRAININGSET=300
ERROR="absolute"

for batchsize in "${BATCHSIZES[@]}"
do
	for dataset in "${DATASETS[@]}"
	do
		for workload in "${WORKLOADS[@]}"
		do
			for samplesize in "${SAMPLESIZES[@]}"
			do
				for variant in "${VARIANTS[@]}"
				do
					echo "Running workload $workload on $dataset (D=$DIM), samplesize:$samplesize, variant:$variant."
					postgres -d $PGDATA 2>/dev/null &
					PID=$!
					sleep 1
					python runExperiment.py	--dbname=mheimel --dataset=$dataset --dimensions=$DIM \
											--workload=$workload --queries=$QUERIES --samplesize=$samplesize \
											--error=$ERROR --optimization=$variant --trainqueries=$TRAININGSET \
											--log=$LOG --repetitions=$REPETITIONS --batchsize=$batchsize
					kill -9 $PID
					sleep 1
				done
				done
		done
	done
done