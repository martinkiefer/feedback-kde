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

def createModel(table, dimensions, samplefile):
  sys.stdout.write("\tRebuilding estimator ... ")
  sys.stdout.flush()
  # Build the initial model.
  query = "ANALYZE %s(" % table
  for i in range(1, dimensions + 1):
    if (i>1):
      query += ", c%i" % i
    else:
      query += "c%i" %i
  query += ");"
  cur.execute(query)
  if samplefile:
    # Update the sample and reset the bandwidth.
    cur.execute("SELECT kde_import_sample('%s', '%s');" % (table, samplefile))
    cur.execute("SELECT kde_reset_bandwidth('%s');" % table)
  print "done!"

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--port", action="store", type=int, default=5432, help="Port of the postmaster.")
parser.add_argument("--model", action="store", choices=["stholes", "kde_heuristic", "kde_adaptive_rmsprop","kde_adaptive_vsgd","kde_batch", "kde_optimal"], default="none", help="Which model should be used?")
parser.add_argument("--modelsize", action="store", type=int, help="How many rows should the generated model sample?")
parser.add_argument("--error", action="store", choices=["absolute", "relative"], default="absolute", help="Which error metric should be optimized / reported?")
parser.add_argument("--samplefile", action="store", help="Which samplefile should be used?")
parser.add_argument("--train_workload", action="store", help="File containing the training queries")
parser.add_argument("--test_workload", action="store", help="File containing the test queries")
parser.add_argument("--gpu", action="store_true", help="Use the graphics card for the experiment.")
parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
parser.add_argument("--log", action="store", help="Where to append the experimental results?")
parser.add_argument("--logbw", action="store_true", help="Use logarithmic bandwidth representation.")
args = parser.parse_args()


# If not specified, use default filenames.
if args.samplefile:
  sample_filename = args.samplefile
else:
  sample_filename = "/tmp/sample_%s.csv" % args.dbname
if args.train_workload:
  trainworkload_filename = args.train_workload
else:
  trainworkload_filename = "/tmp/train_queries_%s.log" % args.dbname
if args.test_workload:
  testworkload_filename = args.test_workload
else:
  testworkload_filename = "/tmp/test_queries_%s.log" % args.dbname

# Error log file that we will use.
error_log = "/tmp/error_%s.log" % args.dbname

# Open a connection to postgres.
conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (args.dbname, args.port))
conn.set_session('read uncommitted', autocommit=True)
cur = conn.cursor()


if not args.debug:
  cur.execute("SET kde_debug TO false;")

# Extract table name and dimensionality from the query file name.
f = open(testworkload_filename)
query = f.readline()
m = re.search("FROM (.+) WHERE", query)
table = m.groups()[0]
m = re.match(".+([0-9]+)", table)
dimensions = int(m.groups()[0])
f.close()
if (args.model == "stholes"):
   modelsize = args.modelsize
else:
   # Extract the modelsize from the sample file.
   with open(sample_filename) as myfile:
      modelsize = sum(1 for line in myfile)

# Remove all traces of previous experiments. 
cur.execute("DELETE FROM pg_kdemodels;");
cur.execute("DELETE FROM pg_kdefeedback;");
# Also count the number of rows in the table.
cur.execute("SELECT COUNT(*) FROM %s;" % table)
nrows = int(cur.fetchone()[0])

# Set the requested error metric.
if (args.error == "relative"):
    cur.execute("SET kde_error_metric TO SquaredRelative;")
elif (args.error == "absolute"):
    cur.execute("SET kde_error_metric TO Quadratic;")
# Set STHoles specific parameters.
if (args.model == "stholes"):
    cur.execute("SET stholes_hole_limit TO %i;" % modelsize)
    cur.execute("SET stholes_enable TO true;")
    createModel(table, dimensions, "")
else:
    # Set KDE-specific parameters.
    cur.execute("SET kde_samplesize TO %i;" % modelsize)
    if not args.gpu:
      cur.execute("SET ocl_use_gpu TO false;")
    cur.execute("SET kde_enable TO true;")

if args.logbw:
    cur.execute("SET kde_bandwidth_representation TO Log;")

# Initialize the training phase.
if (args.model == "kde_batch"):
    # Drop all existing feedback and start feedback collection.
    cur.execute("SET kde_collect_feedback TO true;")
elif (args.model == "kde_adaptive_rmsprop"):
    # Build an initial model that is being trained.
    cur.execute("SET kde_enable_adaptive_bandwidth TO true;")
    cur.execute("SET kde_minibatch_size TO 10;")
    cur.execute("SET kde_online_optimization_algorithm TO rmsprop;")
    createModel(table, dimensions, sample_filename)
elif (args.model == "kde_adaptive_vsgd"):
    # Build an initial model that is being trained.
    cur.execute("SET kde_enable_adaptive_bandwidth TO true;")
    cur.execute("SET kde_minibatch_size TO 10;")
    cur.execute("SET kde_online_optimization_algorithm TO vSGDfd;")
    createModel(table, dimensions, sample_filename)

# Run the training workload.
trainqueries = 0
if (args.model == "kde_batch" or args.model == "stholes" or args.model == "kde_adaptive_vsgd" or args.model == "kde_adaptive_rmsprop"):
  sys.stdout.write("\tRunning training queries ... ")
  sys.stdout.flush()
  f = open(trainworkload_filename)
  finished_queries = 0
  for query in f:
    trainqueries += 1
    cur.execute(query)
  print "done"

if (args.model == "kde_batch"):
    cur.execute("SET kde_collect_feedback TO false;") # We don't need further feedback collection.    
    cur.execute("SET kde_enable_bandwidth_optimization TO true;")
    cur.execute("SET kde_optimization_feedback_window TO %i;" % trainqueries)
if (args.model != "kde_adaptive_rmsprop" and args.model != "kde_adaptive_vsgd" and args.model != "stholes"):
    createModel(table, dimensions, sample_filename)
# If this is the optimal estimator, we need to compute the PI bandwidth estimate.
if (args.model == "kde_optimal"):
  sys.stdout.write("\tImporting data sample into R ... ")
  sys.stdout.flush()
  m = robjects.r['read.csv']('%s' % sample_filename)
  sys.stdout.write("done!\n\tCalling SCV bandwidth estimator ... ")
  sys.stdout.flush()
  ks = importr("ks")
  bw = robjects.r['diag'](robjects.r('Hscv.diag')(m))
  bw_array = 'ARRAY[%f' % bw[0]
  for v in bw[1:]:
    bw_array += ',%f' % v
  bw_array += ']'
  cur.execute("SELECT kde_set_bandwidth('%s',%s);" % (table, bw_array))
  print "done!"

sys.stdout.write("\tRunning experiment ... ")
sys.stdout.flush()

# Count the number of queries
with open(testworkload_filename) as myfile:
   queries = sum(1 for line in myfile)

if args.logbw:
   args.model += "_log"

# And run the experiments.
f = open(testworkload_filename)
finished_queries = 0
allrows = 0
flog = open(args.log, "a+")
for linenr, line in enumerate(f):
  cur.execute("EXPLAIN ANALYZE %s" % line)
  for row in cur:
    text = row[0]
    if "Seq Scan" in text:
      m = re.match(".+rows=([0-9]+).+rows=([0-9]+).+", text)              
      if not m:
         continue
      error = abs(int(m.group(1)) - int(m.group(2)))                      
      flog.write("_;_;_;%s;%i;%i\n" % (args.model, modelsize, error))
print "done!"
f.close()
flog.close()
conn.close()
