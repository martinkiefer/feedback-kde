#!/usr/bin/env python

# -*- coding: utf-8 -*-
import argparse
import csv
import itertools
import math
import numpy as np
import os
import scipy
import sqlite3
import sys
import time

from numpy import random
from scipy import stats 

database_accesses = 0
def run(cur, template, parameters):
   global database_accesses
   database_accesses += 1
   return cur.execute(template, parameters)

written_queries = 0
def writeQuery(csvwriter, parameters, selectivity):
   global written_queries
   written_queries += 1
   parameters.append(selectivity)
   csvwriter.writerow(parameters)

# Classes generating data centers (Data,Uniform,Gauss)
class DataCenterGenerator:
    def __init__(self, data):
        # Fetch the dataset into memory.
        self.data = data
        self.rows = len(data)

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

def parseData(data_file):
   # Open the data file and read the first line.
   data = []   
   with open(data_file, "r") as f:
      reader = csv.reader(f, delimiter='|')
      for row in reader:
         data.append([float(x) for x in row])
   return data

def loadData(cur, data):
   dim = len(data[0])
   # Create the table.
   query = "CREATE TABLE _d_("
   for i in range(dim):
      if (i > 0):
         query += ", "
      query += "c%i FLOAT" % (i + 1)
   query += ");"
   cur.execute(query)
   # Run the bulk insert.
   query = "INSERT INTO _d_("
   for i in range(dim):
      if (i > 0):
         query += ", "
      query += "c%i" % (i + 1)
   query += ") VALUES ("
   for i in range(dim):
      if (i > 0):
         query += ", "
      query += "?"
   query += ");"
   cur.executemany(query, data)

def printState(output_file_name, target_queries, total_time):
  global database_accesses
  global written_queries
  # Clear the current line
  sys.stdout.write("\r\33[2K")
  sys.stdout.flush()
  sys.stdout.write("\r\tGenerated %i queries for %s, %i remaining (%.2f queries/s, %.2f database accesses / query)" \
                      % (written_queries, output_file_name, target_queries - written_queries,                       \
                         (written_queries / float(total_time)), (database_accesses / float(written_queries))))

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", action="store", required=True, help="CSV File containing the input data.")
parser.add_argument("--selectivity", action="store", default=0.01, type=float, help="Target selectivity for the created query workload.")
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
output_file = args.output
mcenter = args.mcenter
mrange = args.mrange

output_file_name = os.path.basename(output_file)

data = parseData(args.data)
columns = len(data[0])
rows = len(data)

# Extract minimum and maximum values.
low = np.amin(data, axis=0)
high = np.amax(data, axis=0)
ranges = zip(low, high)

# Initialize the SQLite database.
conn = sqlite3.connect(":memory:", isolation_level=None)
cur = conn.cursor()
loadData(cur, data)

# Build a composite index to accelerate our lookups.
query = "CREATE INDEX d_idx ON _d_("
for i in range(0, columns):
   if i>0:
      query += ", "
   query += "c%i" % (i + 1)
query += ")"
cur.execute(query)
conn.commit()

# Build the template to query the database.
template = "SELECT count(*) FROM _d_ WHERE "
for i in range(0, columns):
   if i>0:
      template += "AND "
   template += "c%d>? AND c%d<? " % (i + 1, i + 1)

# Prepare the data center generators 
centers = []
generator = None
if (mcenter == "Data"):
   generator = DataCenterGenerator(data)
elif (mcenter == "Uniform"):
   generator = UniformDataCenterGenerator(low, high, columns)
elif (mcenter == "Gauss"):
   generator = GaussCenterGenerator(low, high, clusters, columns, sigma)
else:
   print("Not yet implemented")

with open(args.output, "w") as csvfile:
   writer = csv.writer(csvfile, delimiter='|')

   # Build volume queries.        
   if (mrange == "Volume"): 
      bounds = []
      edge = 1
      vol = 1
      for r in ranges:
         vol *= (r[1] - r[0])
         edge *= math.pow((r[1] - r[0]), 1.0/columns)
      edge *= math.pow(target_selectivity, 1.0/columns)
      while (written_queries < args.queries):
         r = random.random_sample(columns)
         c = generator.getNextCenter()
         # Identify the selectivity of this query.
         parameters = createBoundsList(c - 0.5 * edge * r, c + 0.5 * edge * r)
         run(cur, template, parameters)
         selectivity = cur.fetchone()[0] / float(rows)
         writeQuery(writer, parameters, selectivity)
  
   # Build tuple queries.
   elif (mrange == "Tuples"):    
      start_time = time.time()
      last_print_time = time.time()

      # Build an acceptance distribution.
      acc = scipy.stats.norm(loc = target_selectivity, scale = 0.5*target_tolerance)
      acc_norm = 1.0

      while (written_queries < args.queries):
         c = generator.getNextCenter()

         # Figure out how much space we have until we hit the boundary.
         ranges = 1.1 * np.minimum(c - low, high - c)

         # Now count how many points fall in this region.
         run(cur, template, createBoundsList(c - ranges, c + ranges))
         selectivity = cur.fetchone()[0] / float(rows)

         # Make a check whether we directly accept the query.
         if (random.random() < (acc_norm * acc.pdf(selectivity))):
            writeQuery(writer, createBoundsList(c - ranges, c + ranges), selectivity)
            continue
         
         # If the query region is too small for a binary search, abort.
         if (selectivity < target_selectivity):
            continue

         # Now start the binary search for the optimal query region.
         lower_bound = 0 
         lower_bound_factor = 0 
         upper_bound = selectivity
         upper_bound_factor = 1 
         test_factor = 1
         while lower_bound_factor < upper_bound_factor: 
            test_factor = 0.5 * (lower_bound_factor + upper_bound_factor)
            run(cur, template, createBoundsList(c - (ranges*test_factor), c + (ranges*test_factor)))
            selectivity = cur.fetchone()[0] / float(rows)

            # Check if we accept this query.
            if (random.random() < (acc_norm * acc.pdf(selectivity))):
               writeQuery(writer, createBoundsList(c - (ranges * test_factor), c + (ranges * test_factor)), selectivity)
               break
            elif (selectivity > target_selectivity):
               upper_bound = selectivity 
               upper_bound_factor = test_factor
            else:
               lower_bound = selectivity
               lower_bound_factor = test_factor
         if (written_queries > 0 and time.time() - last_print_time >= 1):
            # Print the current status every second:
            printState(output_file_name, args.queries, time.time() - start_time)
            sys.stdout.flush()
            last_print_time = time.time()
      printState(output_file_name, args.queries, time.time() - start_time)
      sys.stdout.write("\n") 

conn.close()
