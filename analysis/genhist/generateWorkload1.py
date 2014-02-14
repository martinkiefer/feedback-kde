import argparse
import psycopg2
import random
import sys
import time

def usage():
    print "Usage flags: "
    print "\t--output=<filename> / -o=<filename>"
    print "\t\tName of the output sql file."
    print "\t--algorithm=<random/binary/interpolation> / -a=<random/binary/interpolation>"
    print "\t\tWhich algorithm should be used? (Default: random)"


parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--table", action="store", required=True, help="Table for which the query workload will be generated.")
parser.add_argument("--selectivity", action="store", required=True, type=float, help="Target selectivity for the created query workload.")
parser.add_argument("--tolerance", action="store", required=True, type=float, help="Tolerance around the target selectivity.")
parser.add_argument("--queries", action="store", required=True, type=int, help="Number of queries in the target workload.")
parser.add_argument("--output", action="store", required=True, help="Name of the output sql file.")
parser.add_argument("--method", action="store", choices=["random","binary","interpolation"], default="random", help="Which method should be used to construct the queries?")

args = parser.parse_args()

database = args.dbname
table = args.table
target_selectivity = args.selectivity
target_tolerance = args.tolerance
queries = args.queries
output_file = args.output
method = args.method

# Open a connection to postgres.
conn = psycopg2.connect("dbname=%s host=localhost" % database)
cur = conn.cursor()

# First, we need to figure out the dimensionality of this table.
cur.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name='%s'" % table)
columns = cur.fetchone()[0]

# Now fetch the number of rows.
cur.execute("SELECT COUNT(*) FROM %s" % table)
rows = cur.fetchone()[0]

# Now fetch the min and max values.
ranges = []
for i in range(0, columns):
    # Build an index on this column.
    try:
        cur.execute("CREATE INDEX %s_c%i ON %s(c%i)" % (table, i+1, table, i+1))
    except Exception, e:
        conn.reset()
        pass
    # And fetch the min / max values.
    cur.execute("SELECT MIN(c%i), MAX(c%i) FROM %s" % (i+1, i+1, table))
    result = cur.fetchone()
    low = result[0]
    up = result[1]
    width = up - low
    ranges.append([low - 0.05*width, up + 0.05*width])
conn.commit()

# From now on, we will only read.
conn.set_session('read uncommitted', readonly=True, autocommit=True)

# Build the query template.
template = "SELECT count(*) FROM %s WHERE " % table
for i in range(0, columns):
    if i>0:
        template += "AND "
    template += "c%d>%%s AND c%d<%%s " % (i+1, i+1)
    

workload = []

last_len = 0
last_print_time = time.time()

query = [0] * (columns * 2)
while True:
    # Pick a starting point. 
    for i in range(0, columns):
        if method == "random":
            a = ranges[i][0] + random.random()*(ranges[i][1] - ranges[i][0])
            b = ranges[i][0] + random.random()*(ranges[i][1] - ranges[i][0])
            query[2*i] = min(a, b)
            query[2*i+1] = max(a, b)
        else:
            query[2*i] = ranges[i][0] + random.random()*(ranges[i][1] - ranges[i][0])
            query[2*i+1] = ranges[i][1]
    cur.execute(template, query)
    selectivity = cur.fetchone()[0] / float(rows)
    if (selectivity < (target_selectivity - 0.5*target_tolerance)):
      continue  # Move to the next query.
    if (selectivity > (target_selectivity + 0.5*target_tolerance)):
      if method == "random":
          continue
      # Adjust the query to match 
      lower_bound = 0 
      lower_bound_factor = 0 
      upper_bound = selectivity
      upper_bound_factor = 1 
      test_query = list(query)
      while (upper_bound - lower_bound > target_tolerance and (upper_bound_factor - lower_bound_factor) > 0.001 ):
        # Compute the projected factor.
        test_factor = lower_bound_factor + (target - lower_bound) * float(upper_bound_factor - lower_bound_factor)/float(upper_bound - lower_bound)
        #test_factor = 0.5 * (lower_bound_factor + upper_bound_factor)
        # Evaluate the selectivity
        for i in range(0, columns):
          test_query[2*i + 1] = query[2*i] + test_factor * (query[2*i + 1] - query[2*i])
        cur.execute(template, test_query)
        test = cur.fetchone()[0] / float(rows)
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
    workload.append(list(query))
    if (len(workload) == queries): 
      break
    if (time.time() - last_print_time >= 10):
        print "%f queries / second" % ((len(workload_1) - last_len) / float(10))
        last_print_time = time.time()
        last_len = len(workload_1)
conn.close()

# Prepare writing out the result to disk.
template = template.replace("%s", "%f")
template += ";\n"

f = open(output_file, "w")
# Write out the resulting workload.
for query in workload:
    f.write(template % tuple(query))
f.close()
