# -*- coding: utf-8 -*-

from numpy import random
import numpy
import argparse
import moving_common as mc
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--table", action="store", required=True, help="Table for which the workload will be generated.")
parser.add_argument("--queryoutput", action="store", required=True, help="Outputfile for workload queries")
parser.add_argument("--dataoutput", action="store", required=True, help="Table for which the workload will be generated.")
parser.add_argument("--sigma", action="store", required=True, type=float, help="Standard deviation for clusters")
parser.add_argument("--margin", action="store", required=True, type=float, help="Maximum absolute distance for each dimension of a cluster center from the generating point on the line")
parser.add_argument("--clusters", action="store", type=int,required=True,help="Total number of clusters constructed during runtime")
parser.add_argument("--points", action="store", type=int,required=True,help="Total number of data points")
parser.add_argument("--history", action="store", type=int,required=True,help="Maximum number of clusters maintained paralelly")
parser.add_argument("--steps", action="store", type=int,required=True,help="Steps along the line")
parser.add_argument("--dimensions", action="store", type=int,required=True,help="Number of dimensions")
parser.add_argument("--queriesperstep", action="store", type=int,required=True,help="Number of queries per step")
parser.add_argument("--maxprob", action="store", type=float,required=True,help="Maximum probability for the active cluster to be queried")
parser.add_argument("--c1_queries", action="store", type=int,required=True,help="Generated queries from the first cluster center for optimzations.")
parser.add_argument("--c1_output", action="store", required=True, help="Output file for first cluster queries")

args = parser.parse_args()

history = args.history

#Number of points
points = args.points

#Number of clusters
clusters = args.clusters

#Dimensions
dimension = args.dimensions

#Maximum standard deviation
sigma = args.sigma


steps = args.steps
queries_per_step = args.queriesperstep

margin = args.margin
max_prob = args.maxprob

table = "%s_d%i" % (args.table,dimension)
query_output = args.queryoutput
data_output = args.dataoutput

c1_output = args.c1_output
c1_queries= args.c1_queries

#Step 1: Draw a line through the space 
pos_vector, dir_vector = mc.create_line(dimension)
centers = []

#Create cluster centers along the line
for i in range(1,clusters+1):
    base = pos_vector + (dir_vector/clusters) * i

    r = random.random_sample((dimension))    
    center = base + margin - (r * margin*2)  
    
    center[0]= base[0]
    centers.append(center)

#Create data points around the clusters
tuples = []
for c in centers:    
    r = random.normal(0,sigma,(points/clusters,dimension))
    tuples.append(r+c)    

query_template = "SELECT count(*) FROM %s WHERE " % table
for i in range(0, dimension):
    if i>0:
        query_template += "AND "
    query_template += "c%d>%%f AND c%d<%%f " % (i+1, i+1)
query_template += ";\n"

insert_template = "INSERT INTO %s VALUES ( %%i,"  % table
for i in range(0, dimension):
    if i>0:
        insert_template += ", "
    insert_template += "%f"
insert_template += ");\n"

delete_template = "DELETE FROM %s WHERE CL = %%i;\n" % table

output = open(query_output,'wb')   


current = 0
workload = []


file = open(data_output,'wb')
writer = csv.writer(file)

for point in tuples[0]:
    writer.writerow((current,)+tuple(point))
file.close()
    
for i in range(1,steps+1):
    base = pos_vector + (dir_vector/steps) * i    
    
    while(current < clusters and base[0] > centers[current][0]):
        current += 1
        if(i >= history):
            output.write(delete_template % (current-history))
        for point in tuples[current]:
            output.write(insert_template % ((current,)+tuple(point)))  
    if(current == 0):
        progress = 1.0
    else:
        progress = (dir_vector[0]/clusters - (centers[current][0]-base[0]))/(dir_vector[0]/clusters)
    
    #Calculate probabilities for querying each cluster     
    probs = mc.calc_prob(current,progress,max_prob, clusters,max(current-history+1,0))
    
    query_centers = random.choice(clusters, queries_per_step, p=probs)
    
    queries = []
    for i in query_centers:
        c = centers[i]
        x = c+random.normal(0,sigma,(dimension))
        y = c+random.normal(0,sigma,(dimension))

        low = numpy.minimum(x,y)
        high = numpy.maximum(x,y)
        
        output.write(query_template % tuple(mc.createBoundsList(low,high)))
output.close()

output = open(c1_output,'wb')       
#Write the random queries for optizations
c = centers[0]
for i in range(0,c1_queries):    
    x = c+random.normal(0,sigma,(dimension))
    y = c+random.normal(0,sigma,(dimension))
    
    low = numpy.minimum(x,y)
    high = numpy.maximum(x,y)    
    
    output.write(query_template % tuple(mc.createBoundsList(low,high)))    
output.close()             