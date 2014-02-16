import argparse
import random
import sys
import time

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--table", action="store", required=True, help="Table that contains the data.")
parser.add_argument("--queries", action="store", required=True, type=int, help="Number of queries in the target workload.")
parser.add_argument("--output", action="store", required=True, help="Name of the output sql file.")
parser.add_argument("--type", action="store", choices=["half_range","range"], default="range", help="What type of queries should be constructed?")
parser.add_argument("--database", action="store", choices=["postgres", "monetdb"], default="postgres", help="Which database system should be used to construct the queries (MonetDB will be drastically faster)?")
args = parser.parse_args()

# Fetch them.
dbname = args.dbname
table = args.table
queries = args.queries
output_file = args.output
type = args.type
database = args.database

# Open a database connection
if (database == "monetdb"):
    import monetdb.sql  # Requires building and installing the MonetDB python adapter (see MonetDB website for details).
    conn = monetdb.sql.connect(username="monetdb", password="monetdb", hostname="localhost", database=dbname)
elif (database == "postgres"):
    import psycopg2
    conn = psycopg2.connect("dbname=%s host=localhost" % dbname)

cur = conn.cursor()

# Figure out the dimensionality of this table.
if (database == "monetdb"):
    cur.execute("select count(*) from sys.columns where table_id = (SELECT id FROM sys.tables WHERE name='%s')" % table)
elif (database == "postgres"):
    cur.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name='%s'" % table)
columns = cur.fetchone()[0]

# Fetch the minimum / maximum for each column to determine the query range..
ranges = []
for i in range(0, columns):
    cur.execute("SELECT MIN(c%i), MAX(c%i) FROM %s" % (i+1, i+1, table))
    result = cur.fetchone()
    low = result[0]
    up = result[1]
    width = up - low
    ranges.append([low - 0.01*width, up + 0.01*width])
conn.close()

# Generate the query template.
template = "SELECT count(*) FROM %s WHERE " % table
for i in range(0, columns):
    if i>0:
        template += "AND "
    if (type == "range"):
        template += "c%d>%%f AND c%d<%%f " % (i+1, i+1)
    elif (type == "half_range"):
        template += "c%d<%%f " % (i+1)
template += ";\n"
if (type == "range"):
    query = [0] * (columns * 2)
elif (type == "half_range"):
    query = [0] * (columns)

# Generate the random queries and write them to the file.
f = open(output_file, "w")
for i in range(0, queries):
    for j in range (0,columns):
        if (type == "range"):
            a = ranges[j][0] + random.random()*(ranges[j][1] - ranges[j][0])
            b = ranges[j][0] + random.random()*(ranges[j][1] - ranges[j][0])
            query[2*j] = min(a, b)
            query[2*j+1] = max(a, b)
        elif (type == "half_range"):
            query[j] = ranges[j][0] + random.random()*(ranges[j][1] - ranges[j][0])
    f.write(template % tuple(query))
f.close()
