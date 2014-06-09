# -*- coding: utf-8 -*-
"""
Nice little script to run experiments and plot data, sample, karma and queries.
Needs an existing estimator and a two-dimensional data set.
@author: martin
"""

import struct
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import psycopg2
import os
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


pg_folder = "/home/martin/Dokumente/HiWi/current_data"
sample_size = 512                       #Number of points in the sample
table_oid = 16947                     #Table oid to locate the sample file
database_name="xy"                      #Database name
table_name="gen1_d2"                    #Table name
data_sample_size = 10000                #Size of sample used to visualize data
image_folder="/home/martin"             #Put the images in here
number_of_pictures = 15

#Command that is executed before a new diagram is generated
experiment_command = "python /home/martin/Dokumente/HiWi/Repository/feedback-kde/analysis/genhist/runExperiment.py --dbname xy --dataset set1 --dimensions 2 --workload 2 --queries 25 --optimization none --log log --noanalyze --dumpqueries --samplesize 512 --error absolute"
#File to fetch the last executed queries from
query_file = "/tmp/queries.sql"
#File to read the sample from
sample_file = "%s/pg_kde_samples/rel%s_kde.sample" % (pg_folder,table_oid)
#Query that changes the data after the first picture
changing_query = "delete from %s where c1 > 0.25 and c2 > 0.25 and c1 < 0.75 and c2 < 0.75;;" % table_name

ploti = SamplePlotter()    
conn = psycopg2.connect("dbname=%s host=localhost" % database_name)

#Fetch data sample
cur = conn.cursor()
cur.execute("select * from %s order by random() limit %s;" % (table_name,data_sample_size))
data_points = cur.fetchall()
for tuple in data_points:
    ploti.addDataPoint(tuple)
cur.close()

#Create initial picture with no experiment
f = open(sample_file,"rb")
sample_points = []
penalties = []
    
#Read sample file    
for i in range(0,sample_size):
    sample_points.append(struct.unpack("2d",f.read(8*2)))
    
for i in range(0,sample_size):
    penalties.append(struct.unpack('d',f.read(8))[0])
    
for i in range(0,sample_size):
    ploti.addSamplePoint(sample_points[i],penalties[i])

#Read query file      
#q = open(query_file, "r")
#for line in q:
#    ploti.addQuery(parseRectBoundaries(line))
#q.close()
      
f.close();    
 
ploti.plot(0,image_folder)
ploti.clearQueries()
ploti.clearSample()
ploti.clearData();

#Run changing query
cur = conn.cursor()
cur.execute(changing_query)
cur.close();

#Grab new data sample
cur = conn.cursor()
cur.execute("select * from %s order by random() limit %s;" % (table_name,data_sample_size))
data_points = cur.fetchall()
for tuple in data_points:
    ploti.addDataPoint(tuple)
cur.close()
conn.commit()
conn.close()


for j in range(1,number_of_pictures):
    os.system(experiment_command)
    time.sleep(1)
    f = open(sample_file,"rb")
    sample_points = []
    penalties = []
    
    
    for i in range(0,sample_size):
        sample_points.append(struct.unpack("2d",f.read(8*2)))
    
    for i in range(0,sample_size):
        penalties.append(struct.unpack('d',f.read(8))[0])
    
    for i in range(0,sample_size):
        ploti.addSamplePoint(sample_points[i],penalties[i])
        
    q = open(query_file, "r")
    for line in q:
        ploti.addQuery(parseRectBoundaries(line))
    q.close()
    f.close();    
 
    ploti.plot(j,image_folder)
    ploti.clearQueries()
    ploti.clearSample()
    