import argparse
import csv
import inspect
import os
import psycopg2
import random
import sys
import time


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
                

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--dataset", action="store", choices=["forest"], required=True, help="Which dataset should be run?")
parser.add_argument("--dimensions", action="store", required=True, type=int, help="Dimensionality of the dataset?")
parser.add_argument("--maxholes", action="store", required=True, type=int, help="Bucket limit for the histogram")
parser.add_argument("--workload", action="store", choices=["dv","uv","gv","dt","ut","gt"],required=True, help="Which workload should be run?")
parser.add_argument("--queries", action="store", required=True, type=int, help="How many queries from the workload should be run?")
parser.add_argument("--error", action="store", choices=["absolute", "relative","normalized"], default="absolute", help="Which error metric should be reported?")
parser.add_argument("--trainqueries", action="store", type=int, default=0, help="How many queries should be used to train the model?")
parser.add_argument("--log", action="store", required=True, help="Where to append the experimental results?")
parser.add_argument("--dumpqueries", action="store_true",default=False,help="Dump workload queries to /tmp/queries.sql?")
args = parser.parse_args()

# Fetch the arguments.
dbname = args.dbname
dataset = args.dataset
dimensions = args.dimensions
workload = args.workload
queries = args.queries
forgetfirst = args.trainqueries
errortype = args.error
maxholes = args.maxholes
trainqueries = args.trainqueries
dumpqueries = args.dumpqueries
log = args.log

queries += forgetfirst
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

total_volume = 1
for i in range(0, dimensions):    
    cur.execute("SELECT MIN(c%i), MAX(c%i) FROM %s" % (i+1, i+1, table))
    result = cur.fetchone()
    total_volume *= result[1]-result[0]

# Count the number of rows in the table.
cur.execute("SELECT COUNT(*) FROM %s;" % table)
nrows = int(cur.fetchone()[0])

cur.execute("SELECT pg_backend_pid();")
print("PID: %i" % cur.fetchone()[0])

# Now open the query file and select a random query set.
f = open(os.path.join(querypath, queryfile), "r")
for linecount, _ in enumerate(f):
    pass
f.seek(0)
linecount += 1
selected_queries = range(1,linecount)
random.shuffle(selected_queries)
selected_queries = set(selected_queries[0:queries])


cur.execute("SET kde_estimation_quality_logfile TO '/tmp/error.log';")

# Set the optimization strategy.
cur.execute("SET stholes_hole_limit TO %i;" % maxholes)
cur.execute("SET stholes_enable TO true;")

# Trigger the model optimization.
print "Building estimator ..."
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

executed_queries = []
output_cardinalities = []

if(dumpqueries):
    dfile  = open("/tmp/queries.sql", "wb")
    
# Finally, run the experimental queries:
print "Running experiment:"
finished_queries = 0
allrows = 0
for linenr, line in enumerate(f):
    if linenr in selected_queries:
        if(dumpqueries):
            dfile.write(line)
        cur.execute(line)

        if(errortype == "normalized"): 
            card = cur.fetchone()[0]
            #print("Query: %s" % line)
            #print("Card %i" % card)
            executed_queries.append(line)
            output_cardinalities.append(card)
        finished_queries += 1
        sys.stdout.write("\r\tFinished %i of %i queries." % (finished_queries, queries))
        sys.stdout.flush()
print "\ndone!"
if(dumpqueries):
    dfile.close()
f.close()
conn.close()

# Extract the error from the error file.
ifile  = open("/tmp/error.log", "rb")
reader = csv.reader(ifile, delimiter=";")

header = True
selected_col = 0

sum = 0.0
row_count = 0
error_uniform = 0
    
executed_queries = executed_queries[forgetfirst:]
output_cardinalities = output_cardinalities[forgetfirst:]
if(errortype == "normalized"):
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
    elif(forgetfirst > 0):
        forgetfirst -= 1
    else:
        sum += float(row[selected_col])
        #print(executed_queries.pop(0))
        #print("Tuples: %f" % output_cardinalities.pop(0))
        #print("Error: %f" % (float(row[selected_col])*nrows))
        if( errortype == "normalized"): 
            error_uniform += abs((getRectVolume(executed_queries.pop(0))/total_volume * nrows) - output_cardinalities.pop(0))
        row_count += 1
        
if(len(executed_queries) != 0 and errortype == "normalized"):
    raise Exception("We have less error log lines than executed queries. This is most likely the case because one or more queries contained a hyperrectangle with no volume.")
             
if errortype == "absolute":
    error = nrows * sum / row_count
if errortype == "relative":
    error = 100 * sum / row_count
if errortype == "normalized":
    error_abs = nrows * sum / row_count
    error_uniform /= row_count
    error = error_abs / error_uniform
#print("Sum %f" % sum)
#print("Error: %f" % error)
#print("Nrows: %i" % nrows)
#print("Row count: %i" % row_count)
    
# Now append to the error log.
f = open(log, "a")
if os.path.getsize(log) == 0:
    f.write("Dataset;Dimensions;Workload;Queries,Trainingsize;Errortype;Error\n")
f.write("%s;%i;%s;%i,%i;%s;%f\n" % (dataset, dimensions, workload,queries-args.trainqueries, args.trainqueries, errortype, error))
f.close()


                
        
