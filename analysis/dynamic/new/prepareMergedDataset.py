#! /usr/bin/env python
import argparse
import math
import psycopg2
import sys
import re

parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--port", action="store", type=int, default=5432, help="Port of the postmaster.")
parser.add_argument("--queryfile1", action="store", required=True, help="Name of the first dataset.")
parser.add_argument("--queryfile2", action="store", required=True, help="Name of the second dataset.")
parser.add_argument("--separation", action="store", type=float, default=10.0, help="Separation of the two sets in the merged dataset")
parser.add_argument("--equalize", action="store_true", help="Ensure that both tables contribute the same number of rows to the joint table.")
args = parser.parse_args()

# Open a connection to postgres.
conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (args.dbname, args.port))
conn.set_session('read uncommitted', autocommit=True)
cur = conn.cursor()

# Extract the dataset names from the two query files:
f = open(args.queryfile1, "r")
query = f.readline()
f.close()
m = re.match(".+ FROM ([^\s]+) .+", query)
dataset1 = m.group(1)
f = open(args.queryfile2, "r")
query = f.readline()
f.close()
m = re.match(".+ FROM ([^\s]+) .+", query)
dataset2 = m.group(1)

# We only allow normalized datasets ...
if not "normalized" in dataset1:
  print "Dataset %s is not normalized." % dataset1
  sys.exit(-1)
elif not "normalized" in dataset2:
  print "Dataset %s is not normalized." % dataset2
  sys.exit(-1)
# ... that match in their number of columns.
cur.execute("SELECT count(*) FROM information_schema.columns WHERE table_name='%s'" % dataset1)
set1_cols = cur.fetchone()[0]
if (set1_cols == 0):
  print "Dataset %s does not exist." % dataset1
cur.execute("SELECT count(*) FROM information_schema.columns WHERE table_name='%s'" % dataset2)
set2_cols = cur.fetchone()[0]
if (set2_cols == 0):
  print "Dataset %s does not exist." % dataset2
elif (set1_cols <> set2_cols):
  print "Datasets have different number of columns!"
  sys.exit(-1)

# If we should equalize the contribution of both tables, we need to know
# the number of rows.
if args.equalize:
  cur.execute("SELECT count(*) FROM %s" % dataset1)
  set1_rows = cur.fetchone()[0]
  cur.execute("SELECT count(*) FROM %s" % dataset2)
  set2_rows = cur.fetchone()[0]

# Create the new table name.
if (dataset1 < dataset2):
  table_name = dataset1 + "_" + dataset2
else:
  table_name = dataset2 + "_" + dataset1
table_name = table_name.replace("normalized", "")
# Ensure that the table is deleted.
try:
  cur.execute("DROP TABLE %s" % table_name)
except:
  pass

# Compute the required separation between both datasets:
sep = args.separation / (2 * math.sqrt(set1_cols))

# Prepare the SQL statement to transition the first dataset into the new table.
query = "SELECT "
for i in range(1,set1_cols+1):
  if i > 1:
    query += ", "
  query += "c%i + %f AS c%i" % (i, sep, i)
query += " INTO %s FROM %s" % (table_name, dataset1)
if (args.equalize and set1_rows > set2_rows):
  # This dataset is larger than the second one. We only insert a random sample
  # of size set2_rows:
  query += " ORDER BY RANDOM() LIMIT %i" % set2_rows
cur.execute(query)

# Now transition the second dataset into the new table.
query = "INSERT INTO %s SELECT " % table_name
for i in range(1,set1_cols+1):
  if i > 1:
    query += ", "
  query += "c%i - %f AS c%i" % (i, sep, i)
query += " FROM %s" % dataset2
if (args.equalize and set2_rows > set1_rows):
  # This dataset is larger than the second one. We only insert a random sample
  # of size set2_rows:
  query += " ORDER BY RANDOM() LIMIT %i" % set1_rows
cur.execute(query)

def transformQuery(query, offset, table):
  predicates = query.split("WHERE")[1].split("AND")
  # Build the queries for both the rescaled and the normalized table.
  query = "SELECT count(*) FROM %s WHERE " % table
  first = True
  for predicate in predicates:
    m = re.search("c([1-9])([<>])(-?[0-9]+\.[0-9]+)", predicate)
    pred_column = int(m.group(1))
    pred_op = m.group(2)
    pred_value = float(m.group(3)) + offset
    # Rebuild the query string.
    if not first:
      query += "AND "
    first = False
    query += "c%i%s%f " % (pred_column, pred_op, pred_value)
  query += ";\n"
  return query

# Finally, we need to transition the queries as well
fin = open(args.queryfile1, "r")
fout = open("set1.sql", "w")
for q in fin:
  fout.write(transformQuery(q, sep, table_name))
fout.close()
fin.close()

# Finally, we need to transition the queries as well
fin = open(args.queryfile1, "r")
fout = open("set2.sql", "w")
for q in fin:
  fout.write(transformQuery(q, -sep, table_name))
fout.close()
fin.close()
