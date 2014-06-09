# -*- coding: utf-8 -*-
import argparse
import os
import math
from numpy import random
import numpy
import itertools
import time


#Classes generating data centers (Data,Uniform,Gauss)
class DataCenterGenerator:
    def __init__(self,cur,table,rows):
        self._cur = cur
        self._table = table
        self._rows = rows
        #Might not be working on Monet
        self._cur.execute("SELECT * FROM %s ORDER BY RANDOM()" % self._table)
        
    def getNextCenter(self):
        
        self._cur.scroll(random.randint(0,rows),'absolute')
        return [list(self._cur.fetchone())] 

class UniformDataCenterGenerator:
    def __init__(self, low,high,columns):
        self._low = low
        self._high = high  
        self._columns = columns
        
    def getNextCenter(self):
        return self._low+(self._high-self._low)* random.random_sample((1,self._columns))
            
class GaussCenterGenerator:
    def __init__(self, low, high, clusters,columns,sigma):
        self._columns = columns
        self._low = low
        self._high = high 
        self._next= 0
        self._clustercenters = random.random_sample((clusters,columns))
        self._sigma = sigma
        
        
    def getNextCenter(self):
        center = self._low+(self._high-self._low)*self._clustercenters[self._next]
        self._next = self._next + 1
        if(self._next == len(self._clustercenters)):
            self._next = 0
        for delta in random.normal(0,self._sigma,(1,self._columns)):
            return center+random.normal(0,self._sigma,(1,self._columns))

def createBoundsList(min,max):
    result = []
    for x,y in itertools.izip(min[0],max[0]):
        result.append(x)
        result.append(y)
    return result

  
# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--table", action="store", required=True, help="Table for which the workload will be generated.")
parser.add_argument("--selectivity", action="store", required=True, type=float, help="Target selectivity for the created query workload.")
parser.add_argument("--mcenter", action="store", choices=["Data","Uniform","Gauss"],required=True, help="Mechanism choosing the center of range queries")
parser.add_argument("--mrange", action="store", choices=["Volume","Tuples"],required=False, help="Mechanism determining the width of the range queries")
parser.add_argument("--clusters", action="store", type=int,default=100,help="Number of clusters for gaussian center mechanism")
parser.add_argument("--sigma", action="store",type=float,default=25, help="Standard deviation for gaussian center mechanism")
parser.add_argument("--tolerance", action="store", default=0.1, type=float, help="Tolerance around the target selectivity.")
parser.add_argument("--queries", action="store", required=True, type=int, help="Number of queries in the target workload.")
parser.add_argument("--output", action="store", required=True, help="Name of the output sql file.")
parser.add_argument("--method", action="store", choices=["random"], default="random", help="Which method should be used to construct the queries?")
parser.add_argument("--database", action="store", choices=["postgres", "monetdb"], default="postgres", help="Which database system should be used to construct the queries (MonetDB will be drastically faster)?")
args = parser.parse_args()

# Fetch them.
dbname = args.dbname
table = args.table
target_selectivity = args.selectivity
target_tolerance = args.tolerance
sigma = args.sigma
clusters = args.clusters
queries = args.queries
output_file = args.output
method = args.method
mcenter = args.mcenter
mrange = args.mrange
database = args.database

output_file_name = os.path.basename(output_file)

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


ranges = []
low = []
high =  []
for i in range(0, columns):
    # Fetch the minimum / maximum for each column to determine the query range.
    # and calculate the edge of the hypercube incrementally.

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
    low.append(result[0])
    high.append(result[1])  
    ranges.append([result[0], result[1] ])

low = numpy.array(low)
high =  numpy.array(high)
centers = []

cur.close()
conn.commit()
# Tell Postgres that we don't need transaction support.
if (database == "postgres"):
    conn.set_session('read uncommitted', readonly=True, autocommit=True)

generator = None

if(mcenter == "Data"):
    generator = DataCenterGenerator(conn.cursor(),table,rows)

elif(mcenter == "Uniform"):
    generator = UniformDataCenterGenerator(low,high,columns)

elif(mcenter == "Gauss"):
    generator = GaussCenterGenerator(low,high,clusters,columns,sigma)

else:
    print("Not yet implemented")

workload = []
    # Build the query template.
template = "SELECT count(*) FROM %s WHERE " % table
for i in range(0, columns):
    if i>0:
        template += "AND "
    template += "c%d>%%s AND c%d<%%s " % (i+1, i+1)
        
if(mrange == "Volume"): 
    bounds = []
    edge = 1
    vol = 1
    for r in ranges:
        vol *= (r[1] - r[0])
        edge *= math.pow((r[1] - r[0]), 1.0/columns)
        
    edge *= math.pow(target_selectivity, 1.0/columns)

    i = 0
    while(len(workload) < queries):
        r = random.random_sample((1,columns))
        c = generator.getNextCenter()

        workload.append(createBoundsList(c-0.5 * edge * r, c + 0.5 * edge * r))
       
elif(mrange == "Tuples"):    
    last_query_batch = 0

    last_len = 0
    last_print_time = time.time()
    while(len(workload) < queries):
        c = generator.getNextCenter()

        #Take the maximum range
        range = numpy.minimum(c-low,high-c)
        
        cur = conn.cursor()
        cur.execute(template, createBoundsList(c-range,c+range))
    
        selectivity = cur.fetchone()[0] / float(rows)
        cur.close()
        last_query_batch += 1
        if (selectivity < (target_selectivity - 0.5*target_tolerance)):
            continue
        
        
        lower_bound = 0 
        lower_bound_factor = 0 
        upper_bound = selectivity
        upper_bound_factor = 1 
        test_factor = 1
        if (selectivity < (target_selectivity + 0.5*target_tolerance) and selectivity > (target_selectivity - 0.5*target_tolerance)):
            workload.append(createBoundsList(c-(range*test_factor),c+(range*test_factor)))  
            continue
        while (upper_bound - lower_bound > target_tolerance and (upper_bound_factor - lower_bound_factor) > 0.001 ):
            last_query_batch += 1
            test_factor = 0.5 * (lower_bound_factor + upper_bound_factor)
            cur = conn.cursor()
            cur.execute(template, createBoundsList(c-(range*test_factor),c+(range*test_factor)))
            selectivity = cur.fetchone()[0] / float(rows)
            cur.close()
            
            if (selectivity < (target_selectivity + 0.5*target_tolerance) and selectivity > (target_selectivity - 0.5*target_tolerance)):
                break;
            elif (selectivity > target_selectivity):
                upper_bound = selectivity 
                upper_bound_factor = test_factor
            else:
                lower_bound = selectivity
                lower_bound_factor = test_factor 
        if (selectivity < (target_selectivity + 0.5*target_tolerance) and selectivity > (target_selectivity - 0.5*target_tolerance)):           
            workload.append(createBoundsList(c-(range*test_factor),c+(range*test_factor)))  
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
