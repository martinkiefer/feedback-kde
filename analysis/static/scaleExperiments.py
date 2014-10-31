#! /usr/bin/env python
import argparse
import psycopg2
import re
import sys

# Helper class to keep column statistics.
class ColumnStats:
  def __init__(self, minVal, maxVal, avgVal, stdVal):
    self._min = minVal
    self._max = maxVal
    self._avg = avgVal
    self._std = stdVal
  def min(self):
    return self._min
  def max(self):
    return self._max
  def avg(self):
    return self._avg
  def std(self):
    return self._std

# Define and parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--port", action="store", type=int, default=5432, help="Port of the postmaster.")
parser.add_argument("--queryfile", action="store", required=True, help="File with benchmark queries.")
args = parser.parse_args()

# Extract table name and dimensionality.
m = re.match("(.+)([0-9])_([du][vt])_0\.01\.sql", args.queryfile)
table = m.group(1) + m.group(2)
dimensions = int(m.group(2))
workload = m.group(3)

# Build the names of the scaled and normalized tables.
table_scaled = m.group(1) + "rescaled%i" % dimensions
table_normal = m.group(1) + "normalized%i" % dimensions

# Connect to the database.
conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (args.dbname, args.port))
conn.set_session('read uncommitted', autocommit=True)
cur = conn.cursor()

# Extract the required statistics for rescaling the queries.
query = "SELECT "
for i in range(1, dimensions+1):
  query += "MIN(c%i), MAX(c%i), AVG(c%i), STDDEV(c%i)" % (i,i,i,i)
  if (i < dimensions):
    query += ", "
query += " FROM %s;" % table
cur.execute(query)
result = cur.fetchone()
column = []
for i in range(0, dimensions):
  column.append(ColumnStats(float(result[4*i]), float(result[4*i+1]), float(result[4*i+2]), float(result[4*i+3])))
cur.close()
conn.close()

# Alright, now we can finally rewrite the queries:
f_in = open(args.queryfile, "r")
f_scaled = open("%s_%s_0.01.sql" % (table_scaled, workload), "w")
f_normal = open("%s_%s_0.01.sql" % (table_normal, workload), "w")
for query in f_in:
  # Extract all predicates.
  predicates = query.split("WHERE")[1].split("AND")
  # Build the queries for both the rescaled and the normalized table.
  query_scaled = "SELECT count(*) FROM %s WHERE " % table_scaled
  query_normal = "SELECT count(*) FROM %s WHERE " % table_normal
  first = True
  for predicate in predicates:
    m = re.search("c([1-9])([<>])(-?[0-9]+\.[0-9]+)", predicate)
    pred_column = int(m.group(1)) - 1
    pred_op = m.group(2)
    pred_value = float(m.group(3))
    # Compute the scaled values.
    pred_value_scaled = pred_value - column[pred_column].min()
    pred_value_scaled /= (column[pred_column].max() - column[pred_column].min())
    pred_value_normal = pred_value - column[pred_column].avg()
    pred_value_normal /= column[pred_column].std()
    # Rebuild the query string.
    if not first:
      query_scaled += "AND "
      query_normal += "AND "
    first = False
    query_scaled += "c%i%s%f " % (pred_column + 1, pred_op, pred_value_scaled)
    query_normal += "c%i%s%f " % (pred_column + 1, pred_op, pred_value_normal)
  query_scaled += ";"
  query_normal += ";"
  f_scaled.write("%s\n" % query_scaled)
  f_normal.write("%s\n" % query_normal)
f_in.close()
f_scaled.close()
f_normal.close()
