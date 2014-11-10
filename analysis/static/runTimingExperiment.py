#! /usr/bin/env python
import argparse
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

def createModel(table, dimensions, reuse):
  query = ""
  if reuse:
    sys.stdout.write("\r\tRetraining existing estimator ... ")
    sys.stdout.flush()
    query += "SELECT kde_reset_bandwidth('%s');" % table
  else:
    sys.stdout.write("\r\tBuilding estimator ... ")
    sys.stdout.flush()
    query += "ANALYZE %s(" % table
    for i in range(1, dimensions + 1):
      if (i>1):
        query += ", c%i" % i
      else:
        query += "c%i" %i
    query += ");"
  cur.execute(query)
  print "done!"


# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--port", action="store", type=int, default=5432, help="Port of the postmaster.")
parser.add_argument("--queryfile", action="store", required=True, help="File with benchmark queries.")
parser.add_argument("--queries", action="store", required=True, type=int, help="How many queries from the workload should be used?")
parser.add_argument("--trainqueries", action="store", default=100, type=int, help="How many queries should be used to train the model?")
parser.add_argument("--gpu", action="store_true", help="Use the graphics card.")
parser.add_argument("--model", action="store", choices=["none", "stholes", "kde_heuristic", "kde_adaptive","kde_batch", "kde_optimal"], default="none", help="Which model should be used?")
parser.add_argument("--modelsize", action="store", required=True, type=int, help="How many rows should the generated model sample?")
parser.add_argument("--log", action="store", required=True, help="Where to append the experimental results?")
parser.add_argument("--reuse", action="store_true", help="Don't rebuild the model.")

args = parser.parse_args()

# Fetch the arguments.
queryfile = args.queryfile
queries = args.queries
trainqueries = args.trainqueries
model = args.model
modelsize = args.modelsize
log = args.log

# Open a connection to postgres.
conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (args.dbname, args.port))
conn.set_session('read uncommitted', autocommit=True)
cur = conn.cursor()

# Extract table name, workload, selectivity and dimensionality from the query file name.
queryfilename = ntpath.basename(queryfile)
m = re.match("(.+)_([a-z]+)_(.+).sql", queryfilename)
table = m.groups()[0]
workload = m.groups()[1]
selectivity = m.groups()[2]
m = re.match("(.+)([1-9]+)", table)
dataset = m.groups()[0]
dimensions = int(m.groups()[1])

# Determine the total volume of the given table.
print "Preparing experiment ...",
if not args.reuse:
  cur.execute("DELETE FROM pg_kdemodels;");
sys.stdout.flush()
total_volume = 1
for i in range(0, dimensions):    
    cur.execute("SELECT MIN(c%i), MAX(c%i) FROM %s" % (i+1, i+1, table))
    result = cur.fetchone()
    total_volume *= result[1]-result[0]
# Also count the number of rows in the table.
cur.execute("SELECT COUNT(*) FROM %s;" % table)
nrows = int(cur.fetchone()[0])

# We will now select a random training and test file. For this,
# we first need to find out how many queries there are in total.
f = open(queryfile)
for total_queries, _ in enumerate(f):
    pass
total_queries += 1
# Now we do a quick sanity check.
if (queries > total_queries or trainqueries > total_queries):
    print "Requested more queries than available."
    sys.exit(-1)
# Ok, now select which queries are in the training and which are in the test set.
selected_training_queries = range(1, total_queries)
random.shuffle(selected_training_queries)
selected_training_queries = set(selected_training_queries[0:trainqueries])
selected_test_queries = range(1, total_queries)
random.shuffle(selected_test_queries)
selected_test_queries = set(selected_test_queries[0:queries])
# If we run heuristic or optimal KDE, we do not need a training set.
if (model == "kde_heuristic" or model == "kde_optimal" or model == "none"):
    trainqueries = 0
print "done!"

start = time.time()

print "Building the initial model ..."
# Set STHoles specific parameters.
if (model == "stholes"):
    cur.execute("SET stholes_hole_limit TO %i;" % modelsize)
    cur.execute("SET stholes_enable TO true;")
    createModel(table, dimensions,False)
elif (model <> "none"):
    # Set KDE-specific parameters.
    cur.execute("SET kde_samplesize TO %i;" % modelsize)
    if (args.gpu):
      cur.execute("SET ocl_use_gpu TO true;")
    else:
      cur.execute("SET ocl_use_gpu TO false;")
    cur.execute("SET kde_enable TO true;")
    cur.execute("SET kde_debug TO false;")

# Initialize the training phase.
if (model == "kde_batch"):
    # Drop all existing feedback and start feedback collection.
    cur.execute("DELETE FROM pg_kdefeedback;")
    cur.execute("SET kde_collect_feedback TO true;")
elif (model == "kde_adaptive"):
    # Build an initial model that is being trained.
    cur.execute("SET kde_enable_adaptive_bandwidth TO true;")
    cur.execute("SET kde_minibatch_size TO 10;")
    cur.execute("SET kde_online_optimization_algorithm TO rmsprop;")
    createModel(table, dimensions,args.reuse)

# Now run the training queries.
f.seek(0)
finished_queries = 0
if (trainqueries > 0):
    for linenr, line in enumerate(f):
        if linenr in selected_training_queries:
            cur.execute(line)
            finished_queries += 1
            sys.stdout.write("\r\tFinished %i of %i training queries." % (finished_queries, trainqueries))
            sys.stdout.flush()
    sys.stdout.write("\n")
if (model == "kde_batch"):
    cur.execute("SET kde_collect_feedback TO false;") # We don't need further feedback collection.    
    cur.execute("SET kde_enable_bandwidth_optimization TO true;")
    cur.execute("SET kde_optimization_feedback_window TO %i;" % trainqueries)
if (model != "kde_adaptive" and model != "stholes" and model != "none"):
    createModel(table, dimensions,args.reuse)
# If this is the optimal estimator, we need to compute the PI bandwidth estimate.
if (model == "kde_optimal"):
  print "Extracting sample for offline bandwidth optimization ..."
  cur.execute("SELECT kde_dump_sample('%s', '/tmp/sample_%s.csv');" % (table, args.dbname))
  print "Importing the sample into R ..."
  m = robjects.r['read.csv']('/tmp/sample_%s.csv' % args.dbname)
  print "Calling SCV bandwidth estimator ..."
  ks = importr("ks")
  bw = robjects.r['diag'](robjects.r('Hscv.diag')(m))
  print "Setting bandwidth estimate: ", bw
  bw_array = 'ARRAY[%f' % bw[0]
  for v in bw[1:]:
    bw_array += ',%f' % v
  bw_array += ']'
  cur.execute("SELECT kde_set_bandwidth('%s',%s);" % (table, bw_array))

print "done!"

end_build = time.time()

print "Running experiment ... "
# Reset the error tracking.
# And run the experiments.
executed_queries = []
output_cardinalities = []
f.seek(0)
finished_queries = 0
allrows = 0
for linenr, line in enumerate(f):
    if linenr in selected_test_queries:
        cur.execute(line)
        finished_queries += 1
        sys.stdout.write("\r\tFinished %i of %i queries." % (finished_queries, queries))
        sys.stdout.flush()
print "\ndone!"
f.close()
conn.close()

end_experiment = time.time()

f = open(log, "a+")
if os.path.getsize(log) == 0:
    f.write("Dimensions;Model;GPU;ModelSize;Trainingsize;BuildTime;ExperimentTime\n")
f.write("%i;%s;%s;%i;%i;%i\n" % (dimensions, model, args.gpu, modelsize, int(1000*(end_build - start)), int(1000*(end_experiment - end_build))))
f.close()

print "done!"
