# -*- coding: utf-8 -*-
"""
Nice little script to run experiments and plot data, sample, karma and queries.
Needs an existing estimator and a two-dimensional data set with columns c1 and c2.
@author: martin
"""

import struct
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import psycopg2
import argparse
import random
import time
    
#Read rectangle boundaries from query
def parseRectBoundaries(query):
    first = -1
    last = -1
    lower_bound = -1
    
    boundaries = []
    
    for n,c in enumerate(query):
        if(c == '<' or c == '>'):
            first = n
        elif(c == ' '):
            last = n
            if(first != -1):
                query[first+1:last+1]
                if(lower_bound == -1):
                    boundaries.append(float(query[first+1:last+1]))
                else:
                    boundaries.append(float(query[first+1:last+1]))
                    lower_bound = -1
                first=-1
                last=-1
    return boundaries
                
#Sort by penalty
def sort_penalty(element):
    return element[1]
        
class SamplePlotter:
    sample_points = []
    data_points = []
    query_boundaries = []
    
    #Clear the last queries
    def clearQueries(self):
        self.query_boundaries = []

    #Clear the last sample and penalties
    def clearSample(self):
        self.sample_points = []    
    
    def clearData(self):
        self.data_points = []    
    #Add a query
    def addQuery(self,boundaries):
        self.query_boundaries.append(boundaries)
    
    #Add a sample point
    def addSamplePoint(self,point,penalty):
        self.sample_points.append((point,penalty))
    
    #Add a data point.
    def addDataPoint(self,point):
        self.data_points.append(point);
    
    #Plot and create $image_folder/image$number.png    
    def plot(self,number,image_folder):
        #Plot data points
        for point in self.data_points:
            plt.plot(point[0],point[1], 'b.')
        
        self.sample_points = sorted(self.sample_points, key=sort_penalty)    
       
        min_penalty = min(min(self.sample_points, key=sort_penalty)[1],0)
        max_penalty = max(max(self.sample_points, key=sort_penalty)[1],0)
        dist_min = float(0 - min_penalty)
        dist_max = float(max_penalty)
        
        #Plot sample points
        for point,penalty in self.sample_points:
            if(penalty >= 0 ):
                if(dist_min == 0 ):
                    factor = 0.5
                else:    
                    factor = 0.5 - 0.5 * (float(penalty) / dist_max)
            else:
                factor = 0.5 + 0.5 * (0-float(penalty) / dist_min )
            
            #If you want to change the coloring scheme this is the place to do so
            #Plot the sample point
            plt.plot(point[0],point[1], color=cm.jet(factor), marker="o")
            
        #Plot query rectangles
        currentAxis = plt.gca()
        for b in self.query_boundaries: 
            currentAxis.add_patch(Rectangle((b[0], b[2]), b[1]-b[0], b[3]-b[2], linewidth=0.4,edgecolor="black",fill=False,zorder=25))
        
        #Save and close
        plt.savefig("%s/image%s.png" % (image_folder,number))
        plt.close();
        
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--port", action="store", type=int, default=5432, help="Port of the postmaster.")
parser.add_argument("--query_file", action="store", required=True, help="File to draw queries from.")
parser.add_argument("--table", action="store", required=True, help="Table the script will visualize.")
parser.add_argument("--delete_constraint", action="store",default="", help="Constraint removing tuples from the data set after taking the first picture (Appended after SQL WHERE)")
parser.add_argument("--resolution", action="store", required=True, type=int, help="Number of points to visualize the existing data.")
parser.add_argument("--steps", action="store", required=True, type=int, help="Total number of pictures to create.")
parser.add_argument("--queries_per_step", action="store", required=True, type=int, help="Queries executed before a picture is taken.")
parser.add_argument("--folder", action="store", required=True, help="Folder to drop the pictures in.")
parser.add_argument("--sample_maintenance", action="store", choices=["threshold", "periodic","none"], default="none", help="Desired query based sample maintenance option.")
parser.add_argument("--threshold", action="store", type=float, default=1.0, help="Negative karma limit causing a point to be resampled.")
parser.add_argument("--period", action="store", type=int, default=25, help="Queries until we resample the worst sample point.")
args = parser.parse_args()

query_file = args.query_file
table_name=args.table
delete_constraint =  args.delete_constraint                 
data_sample_size = args.resolution      
image_folder= args.folder               
number_of_pictures = args.steps+1         
queries_per_step = args.queries_per_step  
threshold = args.threshold
period = args.period
sample_maintenance = args.sample_maintenance

#Grab some queries and shuffle them.    
f = open(query_file, "r")
selected_queries = f.read().splitlines()
random.shuffle(selected_queries)
selected_queries = selected_queries[0:queries_per_step*number_of_pictures]

f.close()

ploti = SamplePlotter()    
conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (args.dbname, args.port))

cur = conn.cursor()
#File to read the sample from
cur.execute("SELECT pg_kdemodels.sample_file, pg_kdemodels.rowcount_sample FROM pg_kdemodels INNER JOIN pg_class on pg_kdemodels.table = pg_class.oid where pg_class.relname = '%s';" % table_name)
tup = cur.fetchone()
sample_file = tup[0] 
sample_size = tup[1]
cur.close()

sample_query = "select * from %s order by random() limit %s" % (table_name,data_sample_size)

#Fetch data sample
cur = conn.cursor()
cur.execute("select setseed(0.5);");
cur.execute(sample_query)
data_points = cur.fetchall()
for tuple in data_points:
    ploti.addDataPoint(tuple)
cur.close()
data_points = None

#Create initial picture with no experiment
f = open(sample_file,"rb")
sample_points = []
penalties = []
print "Plot intitial state..."    
#Read sample file    
for i in range(0,sample_size):
    sample_points.append(struct.unpack("2d",f.read(8*2)))

for i in range(0,sample_size):
    penalties.append(struct.unpack('d',f.read(8))[0])
    
for i in range(0,sample_size):
    ploti.addSamplePoint(sample_points[i],penalties[i])
      
f.close();    
 
ploti.plot(0,image_folder)
ploti.clearQueries()
ploti.clearSample()


#Run changing query
#Query that changes the data after the first picture
remove_clause = ""
if delete_constraint != "":
    print "Applying data changes..."
    ploti.clearData();

    #Before we execute the actual query, we reduce the sample 
    cur = conn.cursor()
    cur.execute("set SEED to 0.5")
    cur.execute("with sample as ( %s ) select * from sample where not ( %s )" % (sample_query, delete_constraint))
    
    data_points = cur.fetchall()
    for tuple in data_points:
        ploti.addDataPoint(tuple)
    data_points = None    
    cur.close()
    
    cur = conn.cursor()
    cur.execute("delete from %s where %s" % (table_name, delete_constraint))
    cur.close();
    
conn.commit()
conn.close()




offset = 0
for j in range(1,number_of_pictures):
    print "Executing step %i..." % j
    #We need a new connection every time, as we need the estimator persisted
    conn = psycopg2.connect("dbname=%s host=localhost" % database_name)
    cur = conn.cursor()

    #Set the gpukde options
    cur.execute("SET ocl_use_gpu TO false;")
    if(sample_maintenance == "threshold"):
        cur.execute("SET kde_sample_maintenance_query_propagation TO Threshold;")	
        cur.execute("SET kde_sample_maintenance_threshold TO %s;" % threshold)	
    if(sample_maintenance == "periodic"):
        cur.execute("SET kde_sample_maintenance_query_propagation TO Periodic;")	
        cur.execute("SET kde_sample_maintenance_period  TO %s;" % period )	
    cur.execute("SET kde_estimation_quality_logfile TO '/tmp/error.log';")
    cur.execute("SET kde_debug TO false;")    
    cur.execute("set kde_enable to 1;")
    
    #Execute queries and tell the plotter about it
    for i in range(0,queries_per_step):
        cur.execute(selected_queries[offset + i])
        ploti.addQuery(parseRectBoundaries(selected_queries[offset + i]))
    offset += queries_per_step
    
    #Close the connection
    cur.close()
    conn.commit()
    conn.close()        
    #Give postgres some time to write the sample and penalties
    time.sleep(1)
    
    print "Plotting the state..."    
    f = open(sample_file,"rb")
    sample_points = []
    penalties = []
    
    #Tell the plotter about it
    for i in range(0,sample_size):
        sample_points.append(struct.unpack("2d",f.read(8*2)))
        
    for i in range(0,sample_size):
        penalties.append(struct.unpack('d',f.read(8))[0])
    
    for i in range(0,sample_size):
        ploti.addSamplePoint(sample_points[i],penalties[i])
        
    f.close();    
 
 
    ploti.plot(j,image_folder)
    ploti.clearQueries()
    ploti.clearSample()
    print "Done"


    
