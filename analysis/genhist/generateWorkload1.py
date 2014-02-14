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

tuple = [0] * (columns * 2)
while True:
    # Build a random range.
    for i in range(0, columns):
        a = ranges[i][0] + random.random()*(ranges[i][1] - ranges[i][0])
        b = ranges[i][0] + random.random()*(ranges[i][1] - ranges[i][0])
        #b = a + 0.2*random.random()*(ranges[i][1] - a)
        tuple[2*i] = min(a, b)
        tuple[2*i+1] = max(a, b)
    cur.execute(template, tuple)
    selectivity = cur.fetchone()[0] / float(rows)
    if (selectivity > 0.005 and selectivity < 0.015):
        # This is a workload 1 query.
        workload_1.append(list(tuple))
        if (len(workload_1) == 10): 
            break
    if (time.time() - last_print_time >= 5):
        print "%f queries per second" % ((len(workload_1) - last_len) / float(5))
        last_print_time = time.time()
        last_len = len(workload_1)

conn.close()

# Write out the resulting workload.
for tuple in workload_1:
    print template