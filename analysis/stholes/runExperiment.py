import argparse
import csv
import inspect
import os
import psycopg2
import random
import sys
import time

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--dataset", action="store", choices=["forest"], required=True, help="Which dataset should be run?")
parser.add_argument("--dimensions", action="store", required=True, type=int, help="Dimensionality of the dataset?")
parser.add_argument("--workload", action="store", choices=["dv","uv","gv","dt","ut,","gt,"],required=True, help="Which workload should be run?")
parser.add_argument("--queries", action="store", required=True, type=int, help="How many queries from the workload should be run?")
parser.add_argument("--samplesize", action="store", type=int, default=2400, help="How many rows should the generated model sample?")
parser.add_argument("--error", action="store", choices=["absolute", "relative"], default="absolute", help="Which error metric should be optimized / reported?")
parser.add_argument("--optimization", action="store", choices=["none", "adaptive", "batch_random", "batch_workload"], default="none", help="How should the model be optimized?")
parser.add_argument("--trainqueries", action="store", type=int, default=25, help="How many queries should be used to train the model?")
parser.add_argument("--log", action="store", required=True, help="Where to append the experimental results?")
args = parser.parse_args()

# Fetch the arguments.
dbname = args.dbname
dataset = args.dataset
dimensions = args.dimensions
workload = args.workload
queries = args.queries
samplesize = args.samplesize
errortype = args.error
optimization = args.optimization
trainqueries = args.trainqueries
log = args.log

# Open a connection to postgres.
conn = psycopg2.connect("dbname=%s host=localhost" % dbname)
conn.set_session('read uncommitted', autocommit=True)
cur = conn.cursor()

# Fetch the base path for the query files.
basepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if dataset == "set1":
    querypath = os.path.join(basepath, "set1/queries")
    table = "gen1_d%i" % dimensions
if dataset == "set2":
    querypath = os.path.join(basepath, "set2/queries")
    table = "gen2_d%i" % dimensions
if dataset == "set4":
    querypath = os.path.join(basepath, "set4/queries")
    table = "gen4_d%i" % dimensions
if dataset == "set5":
    querypath = os.path.join(basepath, "set5/queries")
    table = "gen5_d%i" % dimensions
if dataset == "tpch":
    querypath = os.path.join(basepath, "set3/queries")
    table = "tpch_data"
if dataset == "forest":
    querypath = os.path.join(basepath, "forest/queries")
    table = "forest%i" % dimensions
queryfile = "%s_%s.sql" % (table, workload)

if (optimization != "none" and optimization != "adaptive"):
    print "Collecting feedback for experiment:"
    sys.stdout.flush()
    # Fetch the optimization queries.
    if (optimization == "batch_random"):
        optimization_query_file = "%s_%i.sql" % (table, 5)
    elif (optimization == "batch_workload"):
        optimization_query_file = queryfile
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

# Now open the query file and select a random query set.
f = open(os.path.join(querypath, queryfile), "r")
for linecount, _ in enumerate(f):
    pass
f.seek(0)
linecount += 1
selected_queries = range(1,linecount)
random.shuffle(selected_queries)
selected_queries = set(selected_queries[0:queries])

# Set all required options.
cur.execute("SET ocl_use_gpu TO false;")
cur.execute("SET kde_estimation_quality_logfile TO '/tmp/error.log';")
if (errortype == "relative"):
    cur.execute("SET kde_error_metric TO SquaredRelative;")
elif (errortype == "absolute"):
    cur.execute("SET kde_error_metric TO Quadratic;")
cur.execute("SET kde_samplesize TO %i;" % samplesize)
# Set the optimization strategy.
if (optimization == "adaptive"):
    cur.execute("SET kde_enable_adaptive_bandwidth TO true;")
    cur.execute("SET kde_adaptive_bandwidth_minibatch_size TO 5;")
elif (optimization == "batch_random" or optimization == "batch_workload"):
    cur.execute("SET kde_enable_bandwidth_optimization TO true;")
    cur.execute("SET kde_optimization_feedback_window TO %i;" % trainqueries)
cur.execute("SET kde_debug TO false;")
cur.execute("SET kde_enable TO true;")

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
for linenr, line in enumerate(f):
    if linenr in selected_queries:
        cur.execute(line)
        finished_queries += 1
        sys.stdout.write("\r\tFinished %i of %i queries." % (finished_queries, queries))
        sys.stdout.flush()
print "\ndone!"
f.close()
conn.close()

# Extract the error from the error file.
ifile  = open("/tmp/error.log", "rb")
reader = csv.reader(ifile, delimiter=";")

header = True
selected_col = 0

sum = 0.0
row_count = 0

for row in reader:
    if header:
        for col in row:
            if (col.strip().lower() != errortype):
                selected_col += 1
            else:
                break
        if (selected_col == len(row)):
            print "Error-type %s not present in given file!" % errortype
            sys.exit()
        header = False
    else:
        sum += float(row[selected_col])
        row_count += 1

if errortype == "absolute":
    error = nrows * sum / row_count
if errortype == "relative":
    error = 100 * sum / row_count

# Now append to the error log.
f = open(log, "a")
if os.path.getsize(log) == 0:
    f.write("Dataset;Dimensions;Workload;Samplesize;Optimization;Trainingsize;Errortype;Error\n")
f.write("%s;%i;%s;%i;%s;%i;%s;%f\n" % (dataset, dimensions, workload, samplesize, optimization, trainqueries, errortype, error))
f.close()
