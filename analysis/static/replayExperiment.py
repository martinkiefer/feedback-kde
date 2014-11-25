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

#Extracts the hyperrectangle size from a query string
#This function relies on a fixed query format:
# - Upper bound and lower bound for every dimension are specified
# - The upper bound clause for a dimension follows immediately on its lower bound clause
def getRectVolume(query):
    first = -1
    last = -1
    lower_bound = -1
    
    vol = 1
    
    for n,c in enumerate(query):
        if(c == '<' or c == '>'):
            first = n
        elif(c == ' '):
            last = n
            if(first != -1):
                query[first+1:last+1]
                if(lower_bound == -1):
                    lower_bound = float(query[first+1:last+1])
                else:
                    vol *= float(query[first+1:last+1]) - lower_bound
                    lower_bound = -1
                first=-1
                last=-1
    return vol

def createModel(table, dimensions, samplefile):
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
  # Print the bandwidth.
  cur.execute("SELECT kde_get_bandwidth('%s');" % table)
  print "\tInitialized with bandwidth %s." % cur.fetchone()  

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--port", action="store", type=int, default=5432, help="Port of the postmaster.")
parser.add_argument("--model", action="store", choices=["stholes", "kde_heuristic", "kde_adaptive_rmsprop","kde_adaptive_vsgd","kde_batch", "kde_optimal"], default="none", help="Which model should be used?")
parser.add_argument("--error", action="store", choices=["absolute", "relative", "normalized"], default="absolute", help="Which error metric should be optimized / reported?")
parser.add_argument("--samplefile", action="store", help="Which samplefile should be used?")
parser.add_argument("--train_workload", action="store", help="File containing the training queries")
parser.add_argument("--test_workload", action="store", help="File containing the test queries")
parser.add_argument("--gpu", action="store_true", help="Use the graphics card for the experiment.")
parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
args = parser.parse_args()

# Set the input file names.
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
# Extract the modelsize from the sample file.
with open(sample_filename) as myfile:
   modelsize = sum(1 for line in myfile)

# Determine the total volume of the given table.
cur.execute("DELETE FROM pg_kdemodels;");
cur.execute("DELETE FROM pg_kdefeedback;");
sys.stdout.flush()
total_volume = 1
for i in range(0, dimensions):    
    cur.execute("SELECT MIN(c%i), MAX(c%i) FROM %s" % (i+1, i+1, table))
    result = cur.fetchone()
    total_volume *= result[1]-result[0]
# Also count the number of rows in the table.
cur.execute("SELECT COUNT(*) FROM %s;" % table)
nrows = int(cur.fetchone()[0])

print "Initializing model ..."
# Set the requested error metric.
if (args.error == "relative"):
    cur.execute("SET kde_error_metric TO SquaredRelative;")
elif (args.error == "absolute" or args.error == "normalized"):
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

# Initialize the training phase.
if (args.model == "kde_batch"):
    # Drop all existing feedback and start feedback collection.
    cur.execute("DELETE FROM pg_kdefeedback;")
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
train_queries = 0
if (args.model == "kde_batch" or args.model == "kde_adaptive_vsgd" or args.model == "kde_adaptive_rmsprop"):
  with open(trainworkload_filename) as myfile:
    trainqueries = sum(1 for line in myfile)
  f = open(trainworkload_filename)
  finished_queries = 0
  for query in f:
    cur.execute(query)
    finished_queries += 1
    sys.stdout.write("\r\tFinished %i of %i training queries." % (finished_queries, trainqueries))
    sys.stdout.flush()
  print ""

if (args.model == "kde_batch"):
    cur.execute("SET kde_collect_feedback TO false;") # We don't need further feedback collection.    
    cur.execute("SET kde_enable_bandwidth_optimization TO true;")
    cur.execute("SET kde_optimization_feedback_window TO %i;" % trainqueries)
if (args.model != "kde_adaptive_rmsprop" and args.model != "kde_adaptive_vsgd" and args.model != "stholes"):
    createModel(table, dimensions, sample_filename)
# If this is the optimal estimator, we need to compute the PI bandwidth estimate.
if (args.model == "kde_optimal"):
  print "Importing the sample into R ..."
  m = robjects.r['read.csv']('%s' % sample_filename)
  print "Calling SCV bandwidth estimator ..."
  ks = importr("ks")
  bw = robjects.r['diag'](robjects.r('Hscv.diag')(m))
  bw_array = 'ARRAY[%f' % bw[0]
  for v in bw[1:]:
    bw_array += ',%f' % v
  bw_array += ']'
  cur.execute("SELECT kde_set_bandwidth('%s',%s);" % (table, bw_array))

print "Running experiment ... "

# Reset the error tracking.
cur.execute("SET kde_estimation_quality_logfile TO '%s';" % error_log)
# Count the number of queries
with open(testworkload_filename) as myfile:
   queries = sum(1 for line in myfile)

# And run the experiments.
f = open(testworkload_filename)
executed_queries = []
output_cardinalities = []
finished_queries = 0
allrows = 0
for linenr, line in enumerate(f):
  cur.execute(line)
  if (args.error == "normalized"): 
    card = cur.fetchone()[0]
    executed_queries.append(line)
    output_cardinalities.append(card)
  finished_queries += 1
  sys.stdout.write("\r\tFinished %i of %i queries." % (finished_queries, queries))
  sys.stdout.flush()
print ""
f.close()
conn.close()

# Extract the error from the error file.
ifile  = open(error_log, "rb")
reader = csv.reader(ifile, delimiter=";")
header = True
selected_col = 0
sum = 0.0
row_count = 0
error_uniform = 0

if (args.error == "normalized"):
    col_errortype = "absolute"
else:
    col_errortype = args.error

for row in reader:
    if header:
        for col in row:
            if (col.strip().lower() != col_errortype):
                selected_col += 1
            else:
                break
        if (selected_col == len(row)):
            print "Error-type %s not present in given file!" % col_errortype
            sys.exit()
        header = False
    else:
        sum += float(row[selected_col])
        if( args.error == "normalized"): 
            error_uniform += abs((getRectVolume(executed_queries.pop(0))/total_volume * nrows) - output_cardinalities.pop(0))
        row_count += 1
        
if(len(executed_queries) != 0 and args.error == "normalized"):
    raise Exception("We have fewer error log lines than executed queries. This is most likely the case because one or more queries contained a hyperrectangle with no volume.")
             
if args.error == "absolute":
    error = nrows * sum / row_count
if args.error == "relative":
    error = 100 * sum / row_count
if args.error == "normalized":
    error_abs = nrows * sum / row_count
    error_uniform /= row_count
    error = error_abs / error_uniform
    
print "Measured error: %f" % error 
