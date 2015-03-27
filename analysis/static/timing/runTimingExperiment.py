#! /usr/bin/env python
import argparse
import csv
import inspect
import ntpath
import os
import psycopg2
import random
import re
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

def createQuery(table, dimensions):
  query = "SELECT count(*) FROM %s WHERE " % table
  for i in range(1, dimensions+1):
    interval = [random.uniform(-2.0, 2.0) for x in range(2)]
    interval.sort()
    if (i > 1):
      query += " AND "
    query += "c%i > %f AND c%i < %f" % (i, interval[0], i, interval[1])
  return query

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--port", action="store", type=int, default=5432, help="Port of the postmaster.")
parser.add_argument("--dimensions", action="store", type=int, required=True, help="How many dimensions should be used for the experiment?.")
parser.add_argument("--queries", action="store", required=True, type=int, help="How many queries from the workload should be used?")
parser.add_argument("--trainqueries", action="store", default=100, type=int, help="How many queries should be used to train the model?")
parser.add_argument("--gpu", action="store_true", help="Use the graphics card.")
parser.add_argument("--model", action="store", choices=["none", "stholes", "kde_heuristic", "kde_adaptive","kde_batch", "kde_optimal"], default="none", help="Which model should be used?")
parser.add_argument("--modelsize", action="store", required=True, type=int, help="How many rows should the generated model sample?")
parser.add_argument("--log", action="store", required=True, help="Where to append the experimental results?")
parser.add_argument("--reuse", action="store_true", help="Don't rebuild the model.")

args = parser.parse_args()

# Fetch the arguments.
queries = args.queries
trainqueries = args.trainqueries
model = args.model
modelsize = args.modelsize
log = args.log

# Open a connection to postgres.
conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (args.dbname, args.port))
conn.set_session('read uncommitted', autocommit=True)
cur = conn.cursor()

# Activate the internal timing.
cur.execute("SET kde_timing_logfile TO '/tmp/time_%i.log'" % args.port)

# Prepare the table name.
table = "time%i" % args.dimensions

# Determine the total volume of the given table.
if not args.reuse:
  cur.execute("DELETE FROM pg_kdemodels;");

# If we run heuristic or optimal KDE, we do not need a training set.
if (model == "kde_heuristic" or model == "kde_optimal" or model == "none"):
    trainqueries = 0

# Set STHoles specific parameters.
if (model == "stholes"):
    cur.execute("SET stholes_hole_limit TO %i;" % modelsize)
    cur.execute("SET stholes_enable TO true;")
    createModel(table, args.dimensions,False)
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
    createModel(table, args.dimensions, args.reuse)

# Run the training queries.
finished_queries = 0
if (trainqueries>0):
   print "\tRunning training queries ... "
while (finished_queries < trainqueries):
   cur.execute(createQuery(table, args.dimensions))
   finished_queries += 1
if (trainqueries>0):
  print "done!"

if (model == "kde_batch"):
    cur.execute("SET kde_collect_feedback TO false;") # We don't need further feedback collection.    
    cur.execute("SET kde_enable_bandwidth_optimization TO true;")
    cur.execute("SET kde_optimization_feedback_window TO %i;" % trainqueries)
if (model != "kde_adaptive" and model != "stholes" and model != "none"):
    createModel(table, args.dimensions, args.reuse)
# If this is the optimal estimator, we need to compute the PI bandwidth estimate.
if (model == "kde_optimal"):
  print "\tExtracting sample for offline bandwidth optimization ..."
  cur.execute("SELECT kde_dump_sample('%s', '/tmp/sample_%s.csv');" % (table, args.dbname))
  print "\tImporting the sample into R ..."
  m = robjects.r['read.csv']('/tmp/sample_%s.csv' % args.dbname)
  print "\tCalling SCV bandwidth estimator ..."
  ks = importr("ks")
  bw = robjects.r['diag'](robjects.r('Hscv.diag')(m))
  print "\tSetting bandwidth estimate: ", bw
  bw_array = 'ARRAY[%f' % bw[0]
  for v in bw[1:]:
    bw_array += ',%f' % v
  bw_array += ']'
  cur.execute("SELECT kde_set_bandwidth('%s',%s);" % (table, bw_array))

end_build = time.time()

cur.close()
conn.set_session('read uncommitted', autocommit=True, readonly=True)
cur = conn.cursor()

query_time = 0

sys.stdout.write("\tRunning experiment ... ")
sys.stdout.flush()
# Reset the error tracking.
# And run the experiments.
finished_queries = 0
total_runtime = 0
while (finished_queries < queries):
   query = createQuery(table, args.dimensions)
   ts = time.time()
   cur.execute(query)
   te = time.time()
   total_runtime += (te - ts)
   finished_queries += 1
print "done!"
conn.close()
total_runtime *= 1000

# Now aggregate the measurements.
construction_time = 0
estimation_time = 0
maintenance_time = 0
first = True
f = open("/tmp/time_%i.log" % args.port, "r")
for line in f:
   m = re.search("([a-zA-Z ]+): ([0-9]+)", line)
   if m:
      category = m.group(1)
      runtime = int(m.group(2))
      if category == "Estimation" and not first:
         estimation_time += runtime
      elif category == "Model Construction":
         construction_time += runtime
      elif category == "Model Maintenance" and not first:
         maintenance_time += runtime
      first = False
f.close()

f = open(log, "a+")
if os.path.getsize(log) == 0:
    f.write("Dimensions;Model;GPU;ModelSize;ConstructionTime;EstimationTime;MaintenanceTime;TotalRuntime\n")
f.write("%i;%s;%s;%i;%i;%i;%i;%.2f\n" % (args.dimensions, args.model, args.gpu, args.modelsize, construction_time, estimation_time, maintenance_time, total_runtime))
f.close()
