#/bin/bash

LOG="experiments.log"

# First, run the forest workload.
DATASETS=("forest")
DIM="4"
#WORKLOADS=("uv")
WORKLOADS=("dt" "ut" "gt" "dv" "uv" "gv")
SAMPLESIZES=(62 62 62 62)
BATCHSIZES=(5)
#VARIANTS=("adaptive")
VARIANTS=("none" "batch_workload" "adaptive")
QUERIES=1000
TRAININGSET=1000
ERROR="normalized"
PGDATA="/home/jtx/Dokumente/current_data"

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
					/usr/local/pgsql/bin/postgres -D $PGDATA 2>> postgres.log &
					PID=$!
					sleep 3
					python runExperiment.py	--dbname=xy --dataset=$dataset --dimensions=$DIM \
											--workload=$workload --queries=$QUERIES --samplesize=$samplesize \
											--error=$ERROR --optimization=$variant --trainqueries=$TRAININGSET \
											--log=$LOG$workload$variant --forgetfirst 1000
					kill -9 $PID
					sleep 1
				done
				done
		done
	done
done
