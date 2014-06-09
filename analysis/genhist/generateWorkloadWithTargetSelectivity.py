import argparse
import os
import random
import sys
import time

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--table", action="store", required=True, help="Table for which the workload will be generated.")
parser.add_argument("--selectivity", action="store", required=True, type=float, help="Target selectivity for the created query workload.")
parser.add_argument("--tolerance", action="store", required=True, type=float, help="Tolerance around the target selectivity.")
parser.add_argument("--fixed", action="store", required=False, type=int, help="How many (random) columns should be fixed to the whole range per query?")
parser.add_argument("--queries", action="store", required=True, type=int, help="Number of queries in the target workload.")
parser.add_argument("--output", action="store", required=True, help="Name of the output sql file.")
parser.add_argument("--method", action="store", choices=["random","binary","interpolation"], default="random", help="Which method should be used to construct the queries?")
parser.add_argument("--database", action="store", choices=["postgres", "monetdb"], default="postgres", help="Which database system should be used to construct the queries (MonetDB will be drastically faster)?")
args = parser.parse_args()

# Fetch them.
dbname = args.dbname
table = args.table
target_selectivity = args.selectivity
target_tolerance = args.tolerance
queries = args.queries
output_file = args.output
method = args.method
database = args.database
if args.fixed:
    fixed = args.fixed
else:
    fixed = 0

output_file_name = os.path.basename(output_file)

# Open a database connection
if (database == "monetdb"):
    import monetdb.sql  # Requires building and installing the MonetDB python adapter (see MonetDB website for details).
    conn = monetdb.sql.connect(username="monetdb", password="monetdb", hostname="localhost", database=dbname, autocommit=True)
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

# And the total number of rows in the table.
cur.execute("SELECT COUNT(*) FROM %s" % table)
rows = cur.fetchone()[0]

# Fetch the minimum / maximum for each column to determine the query range..
ranges = []
for i in range(0, columns):
    if (database == "postgres"):
        # On postgres, we need indexes to achieve good performance.
        try:
            cur.execute("CREATE INDEX %s_c%i ON %s(c%i)" % (table, i+1, table, i+1))
            conn.commit()
        except Exception, e:
            conn.reset()
            pass
    cur.execute("SELECT MIN(c%i), MAX(c%i) FROM %s" % (i+1, i+1, table))
    result = cur.fetchone()
    low = result[0]
    up = result[1]
    width = up - low
    ranges.append([low - 0.05*width, up + 0.05*width])

conn.commit()
# Tell Postgres that we don't need transaction support.
if (database == "postgres"):
    conn.set_session('read uncommitted', readonly=True, autocommit=True)

# Build the query template.
template = "SELECT count(*) FROM %s WHERE " % table
for i in range(0, columns):
    if i>0:
        template += "AND "
    template += "c%d>%%s AND c%d<%%s " % (i+1, i+1)

cur.close()


workload = []

last_query_batch = 0

last_len = 0
last_print_time = time.time()

query = [0] * (columns * 2)
while True:
    # Check if we want to fix columns.
    fixed_cols = set()
    if fixed > 0:
        for i in range(0, fixed):
            while True:
                col = random.randint(0, columns-1)
                if not col in fixed_cols:
                    fixed_cols.add(col)
                    break;
    # Pick a starting point. 
    for i in range(0, columns):
        if i in fixed_cols:
            query[2*i] = ranges[i][0]
            query[2*i+1] = ranges[i][1]
        else:
            if method == "random":
                a = ranges[i][0] + random.random()*(ranges[i][1] - ranges[i][0])
                b = ranges[i][0] + random.random()*(ranges[i][1] - ranges[i][0])
                query[2*i] = min(a, b)
                query[2*i+1] = max(a, b)
            else:
                query[2*i] = ranges[i][0] + random.random()*(ranges[i][1] - ranges[i][0])
                query[2*i+1] = ranges[i][1]
    cur = conn.cursor()
    cur.execute(template, query)
    last_query_batch += 1
    selectivity = cur.fetchone()[0] / float(rows)
    cur.close()
    if (selectivity < (target_selectivity - 0.5*target_tolerance)):
      continue  # Move to the next query.
    if (selectivity > (target_selectivity + 0.5*target_tolerance)):
      if method == "random":
          continue
      # Adjust the query to match the targeted selectivity range.
      lower_bound = 0 
      lower_bound_factor = 0 
      upper_bound = selectivity
      upper_bound_factor = 1 
      test_query = list(query)
      while (upper_bound - lower_bound > target_tolerance and (upper_bound_factor - lower_bound_factor) > 0.001 ):
        # Compute the projected factor.
        if method == "binary":
            test_factor = 0.5 * (lower_bound_factor + upper_bound_factor)
        if method == "interpolation":
            test_factor = lower_bound_factor + (target_selectivity - lower_bound) * float(upper_bound_factor - lower_bound_factor)/float(upper_bound - lower_bound)
        # Evaluate the selectivity
        for i in range(0, columns):
          if i in fixed_cols:
              test_query[2*i + 1] = query[2*i + 1]
          else:
              test_query[2*i + 1] = query[2*i] + test_factor * (query[2*i + 1] - query[2*i])
        cur = conn.cursor()
        cur.execute(template, test_query)
        last_query_batch += 1
        test = cur.fetchone()[0] / float(rows)
        cur.close()
        if (test < (target_selectivity + 0.5*target_tolerance) and test > (target_selectivity - 0.5*target_tolerance)):
          break;
        elif (test > target_selectivity):
          upper_bound = test 
          upper_bound_factor = test_factor
        else:
          lower_bound = test
          lower_bound_factor = test_factor 
      # This is a workload 1 query.
      query = list(test_query)
    if (selectivity < (target_selectivity + 0.5*target_tolerance) and selectivity > (target_selectivity - 0.5*target_tolerance)):           
            workload.append(list(query))
    if (len(workload) == queries): 
      break
    if (time.time() - last_print_time >= 10):
        print "Generated %i queries for %s, %i remaining (%f queries / second)" \
            % (len(workload), output_file_name, queries - len(workload), \
               (len(workload) - last_len) / float(10))
        last_query_batch = 0
        last_print_time = time.time()
        last_len = len(workload)
conn.close()

# Prepare writing out the result to disk.
template = template.replace("%s", "%f")
template += ";\n"

f = open(output_file, "w")
# Write out the resulting workload.
for query in workload:
    f.write(template % tuple(query))
f.close()
