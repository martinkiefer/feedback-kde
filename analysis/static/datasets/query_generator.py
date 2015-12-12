#!/usr/bin/env python

# -*- coding: utf-8 -*-
import argparse
import csv
import itertools
import math
import numpy as np
import os
import sqlite3
import sys
import time

from numpy import random

total_queries = 0
def run(cur, query, parameters):
   global total_queries
   total_queries += 1
   return cur.execute(query, parameters)

# Classes generating data centers (Data,Uniform,Gauss)
class DataCenterGenerator:
    def __init__(self, cur, rows):
        # Fetch the dataset into memory.
        self.data = []
        for row in cur.execute("SELECT * FROM d;"):
           self.data.append(list(row))
        self.rows = rows

    def getNextCenter(self):
        return self.data[random.randint(0, self.rows)]

class UniformDataCenterGenerator:
    def __init__(self, low,high,columns):
        self._low = low
        self._high = high  
        self._columns = columns
        
    def getNextCenter(self):
        return self._low + (self._high - self._low) * random.random_sample(self._columns)
            
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

def createBoundsList(min_vals, max_vals):
    result = []
    for x,y in itertools.izip(min_vals, max_vals):
        if (x == y):
          x -= 0.0001;
          y += 0.0001;
        result.append(x)
        result.append(y)
    return result

def parseData(c, data_file):
   # Open the data file and read the first line.
   data = []   
   with open(data_file, "r") as f:
      reader = csv.reader(f, delimiter='|')
      for row in reader:
         data.append([float(x) for x in row])
   return data      

def loadData(data):
   dim = len(data[0])
   # Create the table.
   query = "CREATE TABLE d("
   for i in range(dim):
      if (i > 0):
         query += ", "
      query += "c%i FLOAT" % i
   query += ");"
   cur.execute(query)
   # Run the bulk insert.
   query = "INSERT INTO d("
   for i in range(dim):
      if (i > 0):
         query += ", "
      query += "c%i" % i
   query += ") VALUES ("
   for i in range(dim):
      if (i > 0):
         query += ", "
      query += "?"
   query += ");"
   cur.executemany(query, data)
   cur.execute("SELECT count(*) FROM d;")
   return dim
 
# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", action="store", required=True, help="CSV File containing the input data.")
parser.add_argument("--selectivity", action="store", required=True, type=float, help="Target selectivity for the created query workload.")
parser.add_argument("--mcenter", action="store", choices=["Data","Uniform","Gauss"],required=True, help="Mechanism choosing the center of range queries")
parser.add_argument("--mrange", action="store", choices=["Volume","Tuples"],required=False, help="Mechanism determining the width of the range queries")
parser.add_argument("--clusters", action="store", type=int,default=100,help="Number of clusters for gaussian center mechanism")
parser.add_argument("--sigma", action="store",type=float,default=25, help="Standard deviation for gaussian center mechanism")
parser.add_argument("--tolerance", action="store", default=0.01, type=float, help="Tolerance around the target selectivity.")
parser.add_argument("--queries", action="store", required=True, type=int, help="Number of queries in the target workload.")
parser.add_argument("--output", action="store", required=True, help="Name of the output sql file.")
args = parser.parse_args()

# Fetch them.
target_selectivity = args.selectivity
target_tolerance = args.tolerance
sigma = args.sigma
clusters = args.clusters
queries = args.queries
output_file = args.output
mcenter = args.mcenter
mrange = args.mrange

output_file_name = os.path.basename(output_file)

conn = sqlite3.connect(":memory:", isolation_level=None) 
cur = conn.cursor()
data = parseData(cur, args.data)
columns = len(data[0])
rows = len(data)

# Extract minimum and maximum values.
low = np.amin(data, axis=0)
high = np.amax(data, axis=0)
ranges = zip(low, high)

# Now generate the query centers.
centers = []
generator = None

# Check if we need to load the data.
if (mcenter == "Data" or mrange == "Tuples"):
   loadData(data)
   # Build an composite index.
   query = "CREATE INDEX d_idx ON d("
   for i in range(0, columns):
      if i>0:
         query += ", "
      query += "c%i" % i
   query += ")"
   cur.execute(query)
   conn.commit()

if (mcenter == "Data"):
   # Load the data.
   generator = DataCenterGenerator(conn.cursor(), rows)
elif (mcenter == "Uniform"):
   generator = UniformDataCenterGenerator(low, high, columns)
elif (mcenter == "Gauss"):
   generator = GaussCenterGenerator(low, high, clusters, columns, sigma)
else:
   print("Not yet implemented")

workload = []
    # Build the query template.
template = "SELECT count(*) FROM d WHERE "
for i in range(0, columns):
    if i>0:
        template += "AND "
    template += "c%d>? AND c%d<? " % (i, i)
        
if (mrange == "Volume"): 
    bounds = []
    edge = 1
    vol = 1
    for r in ranges:
        vol *= (r[1] - r[0])
        edge *= math.pow((r[1] - r[0]), 1.0/columns)
    edge *= math.pow(target_selectivity, 1.0/columns)

    i = 0
    while (len(workload) < queries):
        r = random.random_sample(columns)
        c = generator.getNextCenter()
        workload.append(createBoundsList(c - 0.5 * edge * r, c + 0.5 * edge * r))
       
elif (mrange == "Tuples"):    

    last_len = 0
    start_time = time.time()
    last_print_time = time.time()

    while (len(workload) < queries):
        c = generator.getNextCenter()

        # Figure out how much space we have until we hit the boundary.
        ranges = 1.1 * np.minimum(c - low, high - c)

        # Now count how many points fall in this region.
        run(cur, template, createBoundsList(c - ranges, c + ranges))
        selectivity = cur.fetchone()[0] / float(rows)

        # If the query region is too small, abort directyl. 
        if (selectivity < (target_selectivity - 0.5 * target_tolerance)):
            continue
	
        lower_bound = 0 
        lower_bound_factor = 0 
        upper_bound = selectivity
        upper_bound_factor = 1 
        test_factor = 1
        if (selectivity < (target_selectivity + 0.5*target_tolerance) and selectivity > (target_selectivity - 0.5*target_tolerance)):
            workload.append(createBoundsList(c - (ranges*test_factor), c + (ranges*test_factor)))  
            continue
        # Run a binary search to find the optimal query region.
        while (upper_bound - lower_bound > target_tolerance and (upper_bound_factor - lower_bound_factor) > 0.001 ):
            test_factor = 0.5 * (lower_bound_factor + upper_bound_factor)
            run(cur, template, createBoundsList(c - (ranges*test_factor), c + (ranges*test_factor)))
            selectivity = cur.fetchone()[0] / float(rows)
            
            if (selectivity < (target_selectivity + 0.5*target_tolerance) and selectivity > (target_selectivity - 0.5*target_tolerance)):
                break;
            elif (selectivity > target_selectivity):
                upper_bound = selectivity 
                upper_bound_factor = test_factor
            else:
                lower_bound = selectivity
                lower_bound_factor = test_factor
        if (selectivity < (target_selectivity + 0.5*target_tolerance) and selectivity > (target_selectivity - 0.5*target_tolerance)):           
            workload.append(createBoundsList(c-(ranges*test_factor),c+(ranges*test_factor)))  
        if (time.time() - last_print_time >= 1):
            # Print the current status every second: 
            for i in range(0,140):
                 sys.stdout.write(" ")
            sys.stdout.write("\r") # Clear the line
            sys.stdout.write("\tGenerated %i queries for %s, %i remaining (%.2f queries/s, %.2f database accesses / query)\r" \
                                  % (len(workload), output_file_name, queries - len(workload), \
                                     (len(workload) / float(time.time() - start_time)), \
                                     (total_queries / float(len(workload)))))
            sys.stdout.flush()
            last_print_time = time.time()
            last_len = len(workload)
    sys.stdout.write("\n")    

# Prepare writing out the result to disk.
template = template.replace("?", "%f")
template += ";\n"

f = open(output_file, "w")
# Write out the resulting workload.
for query in workload:
    f.write(template % tuple(query))
f.close()

conn.close()
