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


def createModel(table, dimensions):
  query = ""
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
parser.add_argument("--modelsize", action="store", required=True, type=int, help="How many rows should the generated model sample?")
parser.add_argument("--log", action="store", required=True, help="Where to append the experimental results?")
args = parser.parse_args()

# Open a connection to postgres.
conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (args.dbname, args.port))
conn.set_session('read uncommitted', autocommit=True)
cur = conn.cursor()

# Prepare the table name.
table = "bikenormalized%i" % args.dimensions

# We want to build a full STHoles model, meaning we have to run as many trainqueries as there are holes:
trainqueries = args.modelsize

# Set STHoles specific parameters.
cur.execute("SET stholes_hole_limit TO %i;" % args.modelsize)
cur.execute("SET stholes_enable TO true;")
createModel(table, args.dimensions)

# Run the training queries.
finished_queries = 0
ts = time.time()
if (trainqueries>0):
   print "\tRunning training queries ... "
while (finished_queries < trainqueries):
   cur.execute(createQuery(table, args.dimensions))
   finished_queries += 1
   sys.stdout.write("\r\t\tFinished %i of %i queries. Took %.2f seconds." % (finished_queries, trainqueries, time.time() - ts))
   sys.stdout.flush()
if (trainqueries > 0):
   print " "

cur.close()
conn.set_session('read uncommitted', autocommit=True, readonly=True)
cur = conn.cursor()

query_time = 0

# Activate the internal timing.
cur.execute("SET kde_timing_logfile TO '/tmp/time_%i.log'" % args.port)

sys.stdout.write("\tRunning experiment ... ")
sys.stdout.flush()
# Reset the error tracking.
# And run the experiments.
finished_queries = 0
total_runtime = 0
while (finished_queries < args.queries):
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

f = open(args.log, "a+")
if os.path.getsize(args.log) == 0:
    f.write("Dimensions;Model;GPU;ModelSize;ConstructionTime;EstimationTime;MaintenanceTime;TotalRuntime\n")
f.write("%i;%s;%s;%i;%i;%i;%i;%.2f\n" % (args.dimensions, "stholes", "false", args.modelsize, construction_time, estimation_time, maintenance_time, total_runtime))
f.close()
