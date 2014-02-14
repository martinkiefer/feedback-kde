import psycopg2
import random
import time

# Open a connection to postgres.
conn = psycopg2.connect("dbname=mheimel user=mheimel host=localhost")
cur = conn.cursor()

table="tpch_data"

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
    

# Prepare container to store the queries for the different workload types.
workload_1 = [] # Queries with selectivity ~ 1%

last_len = 0
last_print_time = time.time()

target = 0.01
tolerance = 0.01

query = [0] * (columns * 2)
while True:
    # Pick a starting point. 
    for i in range(0, columns):
        query[2*i] = ranges[i][0] + random.random()*(ranges[i][1] - ranges[i][0])
        #b = a + 0.2*random.random()*(ranges[i][1] - a)
        query[2*i+1] = ranges[i][1]
    cur.execute(template, query)
    selectivity = cur.fetchone()[0] / float(rows)
    if (selectivity < (target-0.5*tolerance)):
      continue  # Move to the next query.
    if (selectivity > (target+0.5*tolerance)):
      # Now do a extrapolation search to pick a viable query range.
      lower_bound = 0 
      lower_bound_factor = 0 
      upper_bound = selectivity
      upper_bound_factor = 1 
      test_query = list(query)
      while (upper_bound - lower_bound > tolerance and (upper_bound_factor - lower_bound_factor) > 0.001 ):
        # Compute the projected factor.
        test_factor = lower_bound_factor + (target - lower_bound) * float(upper_bound_factor - lower_bound_factor)/float(upper_bound - lower_bound)
        # Evaluate the selectivity
        for i in range(0, columns):
          test_query[2*i + 1] = query[2*i] + test_factor * (query[2*i + 1] - query[2*i])
        cur.execute(template, test_query)
        test = cur.fetchone()[0] / float(rows)
        if (test < (target+0.5*tolerance) and test > (target-0.5*tolerance)):
          break;
        elif (test > target):
          upper_bound = test 
          upper_bound_factor = test_factor
        else:
          lower_bound = test
          lower_bound_factor = test_factor 
      # This is a workload 1 query.
      query = list(test_query)
    workload_1.append(list(query))
    if (len(workload_1) == 10000): 
      break
    if (time.time() - last_print_time >= 5):
        print "%f queries / second" % ((len(workload_1) - last_len) / float(5))
        last_print_time = time.time()
        last_len = len(workload_1)
conn.close()

template = template.replace("%s", "%f")

# Write out the resulting workload.
for query in workload_1:
    print template % tuple(query)
