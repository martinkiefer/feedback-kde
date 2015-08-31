import argparse
import csv
import inspect
import os
import psycopg2
import random
import sys
import time
import math

# Extract the error from the error file.
def extractError():
   ifile  = open("/tmp/error.log", "rb")
   reader = csv.reader(ifile, delimiter=";")

   header = True
   selected_col = -1
   tuple_col = -1
   sum = 0.0
   row_count = 0

   column = 0
   for row in reader:
      if header:
         for col in row:
            if (col.strip().lower() == errortype):
               selected_col = column
            if (col.strip().lower() == "tuples"):
               tuple_col = column
            column = column + 1
         if (selected_col == -1 or tuple_col == -1):
            print "Error-type %s or absolute tuple value not present in given file!" % errortype
            sys.exit()
         header = False
      else:
         local_error = float(row[selected_col])
         if (math.isnan(local_error)):
            continue
         row_count +=1
         sum += local_error 

   error = sum / row_count
   # Now append to the error log.
   f = open(log, "a")
   if os.path.getsize(log) == 0:
      f.write("Dimensions;Workload;Samplesize;Optimization;Trainingsize;Errortype;Error\n")
   f.write("%i;%i;%i;%s;%i;%s;%f\n" % (dimensions, workload, samplesize, optimization, trainqueries, errortype, error))
   f.close()

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--dataset", action="store", choices=["mvt", "mvtc_i","mvtc_id"], required=True, help="Which dataset should be run?")
parser.add_argument("--port", action="store", type=int, default=5432, help="Port of the postmaster.")
parser.add_argument("--dimensions", action="store", required=True, type=int, help="Dimensionality of the dataset?")
parser.add_argument("--samplesize", action="store", type=int, default=2400, help="How many rows should the generated model sample?")
parser.add_argument("--error", action="store", choices=["relative","absolute"], default="relative", help="Which error metric should be optimized / reported?")
parser.add_argument("--optimization", action="store", choices=["heuristic", "adaptive", "stholes"], default="heuristic", help="How should the model be optimized?")
parser.add_argument("--trainqueries", action="store", type=int, default=500, help="How many queries should be used to train the model?")
parser.add_argument("--log", action="store", required=True, help="Where to append the experimental results?")
parser.add_argument("--sample_maintenance", action="store", choices=["prr","car","tkr", "pkr","none"], default="none", help="Desired query based sample maintenance option.")
parser.add_argument("--threshold", action="store", type=float, default=0.01, help="Negative karma limit causing a point to be resampled.")
parser.add_argument("--period", action="store", type=int, default=5, help="Queries until we resample the worst sample point.")
parser.add_argument("--limit", action="store", type=float, default=0.9, help="Decay for karma options.")
args = parser.parse_args()

# Fetch the arguments.
dbname = args.dbname
dataset = args.dataset
dimensions = args.dimensions
workload = 0
samplesize = args.samplesize
errortype = args.error
optimization = args.optimization
trainqueries = args.trainqueries
period = args.period
sample_maintenance = args.sample_maintenance
threshold = args.threshold
log = args.log
limit = args.limit

# Open a connection to postgres.
conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (dbname,args.port))
cur = conn.cursor()

# Fetch the base path for the query files.
basepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if dataset == "mvt":
    conn.set_session('read uncommitted', autocommit=True)
    querypath = os.path.join(basepath, "mvt/queries")
    table = "mvt_d%i" % dimensions
if dataset == "mvtc_i":
    conn.set_session(autocommit=True)
    querypath = os.path.join(basepath, "mvtc_i/queries")
    table = "mvtc_i_d%i" % dimensions
if dataset == "mvtc_id":
    conn.set_session(autocommit=True)
    querypath = os.path.join(basepath, "mvtc_id/queries")
    table = "mvtc_id_d%i" % dimensions
queryfile = "%s.sql" % (table)

if (optimization != "heuristic" and optimization != "adaptive" and optimization != "stholes"):
    print "Collecting feedback for experiment:"
    sys.stdout.flush()
    # Fetch the optimization queries.
    f = open(os.path.join(querypath, optimization_query_file), "r")
    for linecount, _ in enumerate(f):
        pass
    linecount += 1
    selected_queries = range(1,linecount)
    random.shuffle(selected_queries)
    selected_queries = set(selected_queries[0:trainqueries])
    # Collect the corresponding feedback.
    cur.execute("DELETE FROM pg_kdefeedback;")
    cur.execute("SET kde_collect_feedback TO true;")
    f.seek(0)
    finished_queries = 0
    for linenr, line in enumerate(f):
        if linenr in selected_queries:
            cur.execute(line)
            finished_queries += 1
            sys.stdout.write("\r\tFinished %i of %i queries." % (finished_queries, trainqueries))
            sys.stdout.flush()
    cur.execute("SET kde_collect_feedback TO false;")
    f.close()
    print "\ndone!"

# Count the number of rows in the table.
cur.execute("SELECT COUNT(*) FROM %s;" % table)
nrows = int(cur.fetchone()[0])

cur.execute("SELECT pg_backend_pid();")
print cur.fetchone()[0]
# Now open the query file and select a random query set.
f = open(os.path.join(querypath, queryfile), "r")

if(sample_maintenance == "tkr"):
    cur.execute("SET kde_sample_maintenance TO TKR;")
    cur.execute("SET kde_sample_maintenance_karma_threshold TO %s;" % threshold)
    cur.execute("SET kde_sample_maintenance_karma_limit TO %s;" % limit)
if(sample_maintenance == "pkr"):
    cur.execute("SET kde_sample_maintenance TO PKR;")
    cur.execute("SET kde_sample_maintenance_period  TO %s;" % period )
    cur.execute("SET kde_sample_maintenance_karma_limit TO %s;" % limit)
if(sample_maintenance == "car"):
    cur.execute("SET kde_sample_maintenance TO CAR;")
if(sample_maintenance == "prr"):
    cur.execute("SET kde_sample_maintenance TO PRR;")
    cur.execute("SET kde_sample_maintenance_period  TO %s;" % period )    
if(sample_maintenance == "none"):
    cur.execute("SET kde_sample_maintenance TO None;")

# Set all required options.
cur.execute("SET ocl_use_gpu TO false;")
cur.execute("SET kde_estimation_quality_logfile TO '/tmp/error.log';")
if (errortype == "relative"):
    cur.execute("SET kde_error_metric TO SquaredRelative;")
elif (errortype == "absolute"):
    cur.execute("SET kde_error_metric TO Quadratic;")
# Set the optimization strategy.
if (optimization == "adaptive"):
    cur.execute("SET kde_enable TO true;")
    cur.execute("SET kde_enable_adaptive_bandwidth TO true;")
    cur.execute("SET kde_minibatch_size TO 5;")
    cur.execute("SET kde_samplesize TO %i;" % samplesize)
elif (optimization == "batch_random" or optimization == "batch_workload"):
    cur.execute("SET kde_enable TO true;")
    cur.execute("SET kde_enable_bandwidth_optimization TO true;")
    cur.execute("SET kde_optimization_feedback_window TO %i;" % trainqueries)
    cur.execute("SET kde_samplesize TO %i;" % samplesize)
elif (optimization == "heuristic"):
    cur.execute("SET kde_enable TO true;")
    cur.execute("SET kde_samplesize TO %i;" % samplesize)
elif (optimization == "stholes"):
    cur.execute("SET stholes_hole_limit TO %i;" % (samplesize / 2))
    cur.execute("SET stholes_enable TO true;")
cur.execute("SET kde_debug TO false;")

# Trigger the model optimization.
print "Building estimator ...",
sys.stdout.flush()
analyze_query = "ANALYZE %s(" % table
for i in range(1, dimensions + 1):
    if (i>1):
        analyze_query += ", c%i" % i
    else:
        analyze_query += "c%i" %i
analyze_query += ");"
cur.execute(analyze_query)
print "done!"

# Finally, run the experimental queries:
print "Running experiment:"
finished_queries = 0
queries = len(f.readlines())
f.seek(0)
for line in f:
    try:
      cur.execute(line) 
    except psycopg2.DatabaseError:
      print "Database error occured. Terminating."
      print line
      print finished_queries+1

      extractError()
      f.close()
      conn.close()
      sys.exit(-1)
    finished_queries += 1
    sys.stdout.write("\r\tFinished %i of %i queries." % (finished_queries, queries))
    sys.stdout.flush()
print "\ndone!"
f.close()
conn.close()
extractError()
