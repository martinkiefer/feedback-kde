#! /usr/bin/env python
import argparse
from cPickle import load
import csv
import inspect
import ntpath
import os
import psycopg2
import random
import re
import rpy2
import sys
import time

from rpy2.robjects.packages import importr
from rpy2 import robjects

# Helper function to load the query workloads.
def prepareWorkload(args):
    # Filenames used to load / record the generated workloads.
    trainworkload_filename = "/tmp/train_queries_%s.log" % args.dbname
    testworkload_filename = "/tmp/test_queries_%s.log" % args.dbname
    # Lists for the workloads.
    testworkload = []
    trainworkload = []
    if args.replay_experiment: # Load the workload from the specified files:
        f = open(trainworkload_filename)
        for query in f:
            trainworkload.append(query)
        f.close()
        f = open(testworkload_filename)
        for query in f:
            testworkload.append(query)
        f.close()
    else: # Select new testing and training workloads.
        sys.stdout.write("\tPreparing new workload ... ")
        sys.stdout.flush()
        # First, count how many queries are in the queryfile.
        f = open(args.queryfile)
        for queries_in_file, _ in enumerate(f):
            pass
        queries_in_file += 1
        f.close()
        # Do a little sanity check:
        if (queries_in_file < args.trainqueries):
            print "Requested more training queries than available."
            sys.exit(-1)
        if (queries_in_file < args.testqueries):
            print "Requested more testing queries than available."
            sys.exit(-1)
        # Select a few random queries for the training workload.
        selected_training_queries = range(1, queries_in_file)
        random.shuffle(selected_training_queries)
        selected_training_queries = set(selected_training_queries[0:args.trainqueries])
        # Select a few random queries for the test workload.
        selected_test_queries = range(1, queries_in_file)
        random.shuffle(selected_test_queries)
        selected_test_queries = set(selected_test_queries[0:args.testqueries])
        # Now walk through the query file and move the selected queries to the corresponding lists.
        f_train = open(trainworkload_filename, "w")
        f_test = open(testworkload_filename, "w")
        f = open(args.queryfile)
        for linenr, query in enumerate(f):
            if linenr in selected_training_queries:
                trainworkload.append(query)
                f_train.write(query)
            if linenr in selected_test_queries:
                testworkload.append(query)
                f_test.write(query)
        f.close()
        f_train.close()
        f_test.close()
        print "done!"
    return (trainworkload, testworkload)

# Helper function to build the estimator model on the given table.
def analyzeTable(args, table, dimensions):
  sys.stdout.write("\tBuilding estimator ... ")
  sys.stdout.flush()
  ts = time.time()
  # Set the statistics target for all columns
  stat_cnt = 100
  if args.model == "none":
    stat_cnt = 1
  for i in range(1, dimensions + 1):
    query = "ALTER TABLE %s ALTER COLUMN c%i SET STATISTICS %i;" % (table, i, stat_cnt)
    cur.execute(query)
  # Trigger reanalyzing the table.
  query = "ANALYZE %s(" % table
  for i in range(1, dimensions + 1):
    if (i>1):
      query += ", c%i" % i
    else:
      query += "c%i" %i
  query += ");"
  cur.execute(query)
  # For KDE, we want to restore the sample to avoid
  if "kde" in args.model:
    sample_file = "/tmp/sample_%s.csv" % args.dbname
    if args.replay_experiment:
      # Import the sample and make sure the bandwidth is re-optimized.
      cur.execute("SELECT kde_import_sample('%s', '%s');" % (table, sample_file))
      cur.execute("SELECT kde_reset_bandwidth('%s');" % table)
    else:
      # Export the sample.
      cur.execute("SELECT kde_dump_sample('%s', '%s');" % (table, sample_file))
  print "done (took %.2f ms)!" % (1000*(time.time() - ts))

# Define and parse the command line arguments:
parser = argparse.ArgumentParser()
    # Database connectivity arguments.
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--port", action="store", type=int, default=5432, help="Port of the postmaster.")
    # Arguments to specify the experiment.
parser.add_argument("--replay_experiment", action="store_true", help="Should the script [replay] the last recorded experiment?")
parser.add_argument("--queryfile", action="store", help="From which file should we take the test & training queries?.")
parser.add_argument("--testqueries", action="store", type=int, help="How many queries should be used to evalaute the model error?")
parser.add_argument("--trainqueries", action="store", default=100, type=int, help="How many queries should be used to train the model?")
    # Arguments to specify the model.
parser.add_argument("--model", action="store", choices=["none", "postgres", "stholes", "kde_heuristic", "kde_scv", "kde_adaptive", "kde_batch"], default="none", help="Which model should be tested?")
parser.add_argument("--modelsize", action="store", required=True, type=int, help="For KDE: How many rows are used in the underlying sample. For STHoles: How many points are used to build the model?")
parser.add_argument("--logbw", action="store_true", help="Use a logarithmic bandwidth representation.")
    # General arguments
parser.add_argument("--log", action="store", required=True, help="Where to append the experimental results?")

args = parser.parse_args()

# First, open the connection to Postgres.
conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (args.dbname, args.port))
conn.set_session('read uncommitted', autocommit=True)
cur = conn.cursor()

# Make sure that debug mode is deactivated and that all model traces are removed (unless we want to reuse the model):
cur.execute("SET kde_debug TO false;")
cur.execute("SET ocl_use_gpu TO true;")
cur.execute("SET kde_error_metric TO Quadratic;")
cur.execute("DELETE FROM pg_kdefeedback;")

#Uncomment to attach gdb
#cur.execute("select pg_backend_pid();")
#print cur.fetchone()
#time.sleep(20)

# Now prepare the workloads:
(training_workload, testing_workload) = prepareWorkload(args)

# Extract the name of the table, its dimensionality and the workload type
m = re.search("FROM (.+) WHERE", testing_workload[0])
table = m.groups()[0]
cur.execute("SELECT count(*) FROM information_schema.columns WHERE table_name='%s';" % table)
dimensions = int(cur.fetchone()[0])
if args.replay_experiment:
    workload_type = '_' # Marker for replayed workloads.
else:
    # Extract the workload type from the query file name:
    m = re.match("(.+)_([a-z]+)_(.+).sql", ntpath.basename(args.queryfile))
    workload_type = m.groups()[1]

# Remove all existing model traces if we don't reuse the model.
cur.execute("DELETE FROM pg_kdemodels;")
cur.execute("DELETE FROM pg_kdefeedback;")
cur.execute("SELECT pg_stat_reset();")

# Make sure we only run training queries if we have an adaptive model:
if (args.model in ("kde_heuristic", "kde_scv", "postgres", "none")):
    training_workload = []
# Set required KDE model parameters.
if "kde" in args.model:
    # KDE-specific parameters.
    cur.execute("SET kde_samplesize TO %i;" % args.modelsize)
    cur.execute("SET kde_enable TO true;")
    if args.logbw:
        cur.execute("SET kde_bandwidth_representation TO Log;")

# Initialize the training phase for batch, adaptive and stholes:
if (args.model == "kde_batch" ):
    cur.execute("SET kde_collect_feedback TO true;")
elif (args.model == "kde_adaptive"):
    # Build an initial model that is being trained.
    cur.execute("SET kde_enable_adaptive_bandwidth TO true;")
    cur.execute("SET kde_minibatch_size TO 10;")
    cur.execute("SET kde_online_optimization_algorithm TO rmsprop;")
    analyzeTable(args, table, dimensions)
elif (args.model == "stholes"):
    # STHoles-specific parameters.
    cur.execute("SET stholes_hole_limit TO %i;" % (args.modelsize / 2))
    cur.execute("SET stholes_enable TO true;")
    analyzeTable(args, table, dimensions)

# Run the training workload.
if training_workload:
    sys.stdout.write("\tRunning training queries ... ")
    sys.stdout.flush()
    ts = time.time()
    for query in training_workload:
       cur.execute(query)
    print "done (took %.2f ms)!" % (1000*(time.time() - ts))

# Build batch models based on the collected feedback.
if (args.model == "kde_batch"):
    cur.execute("SET kde_collect_feedback TO false;") # We don't need further feedback collection.    
    cur.execute("SET kde_enable_bandwidth_optimization TO true;")
    cur.execute("SET kde_optimization_feedback_window TO %i;" % len(training_workload))
# KDE Adaptive and STHOles have already built their model, now we trigger building for all other models.
if (args.model != "stholes" and args.model != "kde_adaptive"):
    analyzeTable(args, table, dimensions)
# Finally, for SCV models, we use the exported sample to compute the optimal bandwidth via SCV
if (args.model == "kde_scv"):
    sample_file = "/tmp/sample_%s.csvsc" % args.dbname
    sys.stdout.write("\tImporting data sample into R ... ")
    sys.stdout.flush()
    m = robjects.r['read.csv']('%s' % sample_file)
    sys.stdout.write("done!\n\tCalling SCV bandwidth estimator ... ")
    sys.stdout.flush()
    ts = time.time()
    ks = importr("ks")
    bw = robjects.r['diag'](robjects.r('Hscv.diag')(m))
    bw_array = 'ARRAY[%f' % bw[0]
    for v in bw[1:]:
        bw_array += ',%f' % v
    bw_array += ']'
    print "done (took %.2f ms)!" % (1000 * (time.time() - ts))
    cur.execute("SELECT kde_set_bandwidth('%s',%s);" % (table, bw_array))

# Ok, we are all set. Run the experiments!
f = open(args.log, "a+")
sys.stdout.write("\tRunning test queries ... ")
sys.stdout.flush()
ts = time.time()
for query in testing_workload:
    cur.execute("EXPLAIN ANALYZE %s" % query)
    # Extract the true and estimated result row count from the explain result
    for row in cur:
        text = row[0]
        if "Seq Scan" in text:
            m = re.match(".+rows=([0-9]+).+rows=([0-9]+).+", text)
            predicted = int(m.group(1))
            actual = int(m.group(2))
            f.write("%s;%i;%s;%s;%i;%i;%i\n" % (table, dimensions, workload_type, args.model, args.modelsize, predicted, actual))
print "done (took %.2f ms)!" % (1000*(time.time() - ts))

# Clean up
f.close()
conn.close()
