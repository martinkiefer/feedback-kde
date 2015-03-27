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
    sys.stdout.write("\tRetraining existing estimator ... ")
    sys.stdout.flush()
    query += "SELECT kde_reset_bandwidth('%s');" % table
  else:
    sys.stdout.write("\tBuilding estimator ... ")
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
parser.add_argument("--model", action="store", choices=["stholes", "kde_heuristic", "kde_adaptive_rmsprop","kde_adaptive_vsgd","kde_batch", "kde_optimal"], default="none", help="Which model should be used?")
parser.add_argument("--modelsize", action="store", required=True, type=int, help="How many rows should the generated model sample?")
parser.add_argument("--error", action="store", choices=["absolute", "relative"], default="absolute", help="Which error metric should be optimized / reported?")
parser.add_argument("--log", action="store", required=True, help="Where to append the experimental results?")
parser.add_argument("--reuse", action="store_true", help="Don't rebuild the model.")
parser.add_argument("--record", action="store_true", help="If set, the script will store the training & testing workload, actual selectivities, as well as the data sample to files in /tmp.")
parser.add_argument("--gpu", action="store_true", help="Use the graphics card for the experiment.")
parser.add_argument("--logbw", action="store_true", help="Use logarithmic bandwidth representation.")

args = parser.parse_args()

# Fetch the arguments.
queryfile = args.queryfile
queries = args.queries
trainqueries = args.trainqueries
model = args.model
modelsize = args.modelsize
errortype = args.error
log = args.log
uselogbw = args.logbw
usegpu = args.gpu

# Error log file that we will use.
error_log = "/tmp/error_%s.log" % args.dbname

# Open a connection to postgres.
conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (args.dbname, args.port))
conn.set_session('read uncommitted', autocommit=True)
cur = conn.cursor()

cur.execute("SET kde_debug TO false;")

# Extract table name, workload, selectivity and dimensionality from the query file name.
queryfilename = ntpath.basename(queryfile)
m = re.match("(.+)_([a-z]+)_(.+).sql", queryfilename)
table = m.groups()[0]
workload = m.groups()[1]
selectivity = m.groups()[2]
m = re.match("(.+)([1-9]+)", table)
dataset = m.groups()[0]
dimensions = int(m.groups()[1])

if "set1" in table:
   table = table.replace("set1", "set1_")

# Determine the total volume of the given table.
if not args.reuse:
  cur.execute("DELETE FROM pg_kdemodels;");
sys.stdout.flush()
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
if (model == "kde_heuristic" or model == "kde_optimal"):
    trainqueries = 0

# Set the requested error metric.
if (errortype == "relative"):
    cur.execute("SET kde_error_metric TO SquaredRelative;")
elif (errortype == "absolute"):
    cur.execute("SET kde_error_metric TO Quadratic;")
# Set STHoles specific parameters.
if (model == "stholes"):
    cur.execute("SET stholes_hole_limit TO %i;" % modelsize)
    cur.execute("SET stholes_enable TO true;")
    createModel(table, dimensions,False)
else:
    # Set KDE-specific parameters.
    cur.execute("SET kde_samplesize TO %i;" % modelsize)
    if not usegpu:
      cur.execute("SET ocl_use_gpu TO false;")
    cur.execute("SET kde_enable TO true;")
if uselogbw:
    cur.execute("SET kde_bandwidth_representation TO Log;")
    
# Initialize the training phase.
if (model == "kde_batch"):
    # Drop all existing feedback and start feedback collection.
    cur.execute("DELETE FROM pg_kdefeedback;")
    cur.execute("SET kde_collect_feedback TO true;")
elif (model == "kde_adaptive_rmsprop"):
    # Build an initial model that is being trained.
    cur.execute("SET kde_enable_adaptive_bandwidth TO true;")
    cur.execute("SET kde_minibatch_size TO 10;")
    cur.execute("SET kde_online_optimization_algorithm TO rmsprop;")
    createModel(table, dimensions,args.reuse)
elif (model == "kde_adaptive_vsgd"):
    # Build an initial model that is being trained.
    cur.execute("SET kde_enable_adaptive_bandwidth TO true;")
    cur.execute("SET kde_minibatch_size TO 10;")
    cur.execute("SET kde_online_optimization_algorithm TO vSGDfd;")
    createModel(table, dimensions,args.reuse)
# Now run the training queries.
f.seek(0)
finished_queries = 0
if args.record: 
  train_queries_f = open("/tmp/train_queries_%s.log" % args.dbname, "w")
  train_selectivities_f = open("/tmp/train_selectivities_%s.log" % args.dbname, "w")
if (trainqueries > 0):
    sys.stdout.write("\tRunning training queries ... ") 
    sys.stdout.flush()
    for linenr, line in enumerate(f):
        if linenr in selected_training_queries:
            cur.execute(line)
            if args.record: 
               train_queries_f.write(line)
               train_selectivities_f.write("%s\n" % cur.fetchone()[0])
            finished_queries += 1
    print "done!"
if (model == "kde_batch"):
    cur.execute("SET kde_collect_feedback TO false;") # We don't need further feedback collection.    
    cur.execute("SET kde_enable_bandwidth_optimization TO true;")
    cur.execute("SET kde_optimization_feedback_window TO %i;" % trainqueries)
if (model != "kde_adaptive_rmsprop" and model != "kde_adaptive_vsgd" and model != "stholes"):
    createModel(table, dimensions,args.reuse)
# If this is the optimal estimator, we need to compute the PI bandwidth estimate.
if (model == "kde_optimal"):
  sys.stdout.write("\tImporting data sample into R ... ")
  sys.stdout.flush()
  cur.execute("SELECT kde_dump_sample('%s', '/tmp/sample_%s.csv');" % (table, args.dbname))
  m = robjects.r['read.csv']('/tmp/sample_%s.csv' % args.dbname)
  sys.stdout.write("done!\n\tCalling SCV bandwidth estimator ... ")
  sys.stdout.flush()
  ks = importr("ks")
  bw = robjects.r['diag'](robjects.r('Hscv.diag')(m))
  bw_array = 'ARRAY[%f' % bw[0]
  for v in bw[1:]:
    bw_array += ',%f' % v
  bw_array += ']'
  cur.execute("SELECT kde_set_bandwidth('%s',%s);" % (table, bw_array))
  print "\tSetting bandwidth estimate: ", bw

if (args.record):
  train_queries_f.close()
  train_selectivities_f.close()
  # Dump the model sample.
  if (model != "stholes"):
    cur.execute("SELECT kde_dump_sample('%s', '/tmp/sample_%s.csv');" % (table, args.dbname))
  # Prepare to dump the run queries.
  wf = open("/tmp/test_queries_%s.log" % args.dbname, "w")

# And run the experiments.
f.seek(0)
allrows = 0
sys.stdout.write("\tRunning experiment ... ")
sys.stdout.flush()
flog = open(log, "a+")
for linenr, line in enumerate(f):
    if linenr in selected_test_queries:
        cur.execute("EXPLAIN ANALYZE %s" % line)
        for row in cur:
          text = row[0]
          if "Seq Scan" in text:
            m = re.match(".+rows=([0-9]+).+rows=([0-9]+).+", text)
            error = abs(int(m.group(1)) - int(m.group(2)))
            flog.write("%s;%i;%s;%s;%i;%i\n" % (dataset, dimensions, workload, model, modelsize, error))
        if args.record:
          wf.write("%s" % line)

print "done!"
wf.close()
f.close()
flog.close()
conn.close()
    
if args.record:
   wf.close()
