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
parser.add_argument("--model", action="store", choices=["stholes", "kde_heuristic", "kde_adaptive_rmsprop","kde_adaptive_vsgd","kde_batch", "kde_optimal"], default="none", help="Which model should be used?")
parser.add_argument("--modelsize", action="store", required=True, type=int, help="How many rows should the generated model sample?")
parser.add_argument("--error", action="store", choices=["absolute", "relative", "normalized"], default="absolute", help="Which error metric should be optimized / reported?")
parser.add_argument("--log", action="store", required=True, help="Where to append the experimental results?")
parser.add_argument("--reuse", action="store_true", help="Don't rebuild the model.")
parser.add_argument("--record", action="store_true", help="If set, the script will store the training & testing workload, actual selectivities, as well as the data sample to files in /tmp.")

args = parser.parse_args()

# Fetch the arguments.
queryfile = args.queryfile
queries = args.queries
trainqueries = args.trainqueries
model = args.model
modelsize = args.modelsize
errortype = args.error
log = args.log

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
if (model == "kde_heuristic" or model == "kde_optimal"):
    trainqueries = 0
print "done!"

print "Building initial model ..."
# Set the requested error metric.
if (errortype == "relative"):
    cur.execute("SET kde_error_metric TO SquaredRelative;")
elif (errortype == "absolute" or errortype == "normalized"):
    cur.execute("SET kde_error_metric TO Quadratic;")
# Set STHoles specific parameters.
if (model == "stholes"):
    cur.execute("SET stholes_hole_limit TO %i;" % modelsize)
    cur.execute("SET stholes_enable TO true;")
    createModel(table, dimensions,False)
else:
    # Set KDE-specific parameters.
    cur.execute("SET kde_samplesize TO %i;" % modelsize)
    cur.execute("SET ocl_use_gpu TO false;")
    cur.execute("SET kde_enable TO true;")

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
    for linenr, line in enumerate(f):
        if linenr in selected_training_queries:
            cur.execute(line)
            if args.record: 
               train_queries_f.write(line)
               train_selectivities_f.write("%s\n" % cur.fetchone()[0])
            finished_queries += 1
            sys.stdout.write("\r\tFinished %i of %i training queries." % (finished_queries, trainqueries))
            sys.stdout.flush()
    sys.stdout.write("\n")
if (model == "kde_batch"):
    cur.execute("SET kde_collect_feedback TO false;") # We don't need further feedback collection.    
    cur.execute("SET kde_enable_bandwidth_optimization TO true;")
    cur.execute("SET kde_optimization_feedback_window TO %i;" % trainqueries)
if (model != "kde_adaptive_rmsprop" and model != "kde_adaptive_vsgd" and model != "stholes"):
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
  bw_array = 'ARRAY[%f' % bw[0]
  for v in bw[1:]:
    bw_array += ',%f' % v
  bw_array += ']'
  cur.execute("SELECT kde_set_bandwidth('%s',%s);" % (table, bw_array))
if args.record:
  train_queries_f.close()
  train_selectivities_f.close()
  # Print the selected bandwidth.
  if (model != "stholes"):
    cur.execute("SELECT kde_get_bandwidth('%s');" % table)
    print "\tEstimated bandwidth: %s" % cur.fetchone() 
print "done!"

print "Running experiment ... "
if (args.record):
   # Dump the model sample.
   if (model != "stholes"):
     cur.execute("SELECT kde_dump_sample('%s', '/tmp/sample_%s.csv');" % (table, args.dbname))
   # Prepare to dump the run queries.
   wf = open("/tmp/test_queries_%s.log" % args.dbname, "w")
# Reset the error tracking.
cur.execute("SET kde_estimation_quality_logfile TO '%s';" % error_log)
# And run the experiments.
executed_queries = []
output_cardinalities = []
f.seek(0)
finished_queries = 0
allrows = 0
for linenr, line in enumerate(f):
    if linenr in selected_test_queries:
        cur.execute(line)
        if args.record: wf.write(line)
        if (errortype == "normalized"): 
            card = cur.fetchone()[0]
            executed_queries.append(line)
            output_cardinalities.append(card)
        finished_queries += 1
        sys.stdout.write("\r\tFinished %i of %i queries." % (finished_queries, queries))
        sys.stdout.flush()
print "\ndone!"
f.close()
conn.close()

print "Extracting error information ... ",
# Extract the error from the error file.
ifile  = open(error_log, "rb")
reader = csv.reader(ifile, delimiter=";")
header = True
selected_col = 0
sum = 0.0
row_count = 0
error_uniform = 0

if (errortype == "normalized"):
    col_errortype = "absolute"
else:
    col_errortype = errortype

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
        if( errortype == "normalized"): 
            error_uniform += abs((getRectVolume(executed_queries.pop(0))/total_volume * nrows) - output_cardinalities.pop(0))
        row_count += 1
        
if(len(executed_queries) != 0 and errortype == "normalized"):
    raise Exception("We have fewer error log lines than executed queries. This is most likely the case because one or more queries contained a hyperrectangle with no volume.")
             
if errortype == "absolute":
    error = nrows * sum / row_count
if errortype == "relative":
    error = 100 * sum / row_count
if errortype == "normalized":
    error_abs = nrows * sum / row_count
    error_uniform /= row_count
    error = error_abs / error_uniform
    
# Now append to the error log.
f = open(log, "a+")
if os.path.getsize(log) == 0:
    f.write("Dataset;Dimension;Workload;Selectivity;Model;ModelSize;Trainingsize;Errortype;Error\n")
f.write("%s;%i;%s;%s;%s;%i;%i;%s;%f\n" % (dataset, dimensions, workload, selectivity, model, modelsize, trainqueries, errortype, error))
f.close()
print "done!"

if args.record:
   wf.close()
   raw_input("Press Enter to continue.")
