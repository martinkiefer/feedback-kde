# -*- coding: utf-8 -*-

from numpy import random
import numpy
import argparse
import moving_common as mc
import csv

#How does this work?
#Step 1: A line is drawn through the R^dimension with position vector 0^$dimensions and direction vector 1^$dimensions
#The base coordinate for a cluster i \in [1,...$clusters] is calculated by  1^$dimensions / $steps * i 

#Step 2: Generate $clusters centers from the base coordinates
#The zeroth dimension of the base vector is kept as a linear time coordinate.
#The other dimensions are drawn uniformly out of the intervall [base_i- $margin,base_i + $margin] 

#Step 3: Create points for each clusters
#$points elements are created for every cluster by sampling from a gaussian distribution with standard deviation $sigma

#Step 4: The first $history clusters are persisted in $dataoutput

#Step 5: Switch the coordinates of the position vector
#Position vector = base coordinate of the last history cluster persisted
#Direction vector = Picked such that pos_vector + dir_vector * $steps = 1^$dimensions

#Step 6: For every step in [1 ... $steps]
#Get the current point on the line by calculating pos_vector + dir_vector * step
#The current cluster is the first one with the zeroth coordinate >= the current point.
#If the current cluster changes the last cluster is removed in "insert_delete" mode.
#Create $points*($cluster-$history)/$steps points for the current cluster
#Create $queriesperstep queries: 
#The difference from the current point to the last cluster on the zeroth dimension
#normalized by the difference between last and current cluster is calculated.
#The probability of queries hitting the current cluster is calculated by rel_difference * $maxprob.
#The "closer" we get to the current cluster, the more queries will hit it.
#The remaining probability mass is distributed among all others by assigning the remainining probability * $max_prob to the next recent cluster
#This is done recursively, the oldest existing cluster gets the remains.

#It is recommended to make $steps a multiple of $history-$clusters to make sure every cluster gets exactly $points members

parser = argparse.ArgumentParser()
parser.add_argument("--table", action="store", required=True, help="Table for which the workload will be generated.")
parser.add_argument("--queryoutput", action="store", required=True, help="Outputfile for workload queries")
parser.add_argument("--dataoutput", action="store", required=True, help="File to dump the initial clusters in.")
parser.add_argument("--sigma", action="store", required=True, type=float, help="Standard deviation for clusters")
parser.add_argument("--margin", action="store", required=True, type=float, help="Maximum absolute distance for each dimension of a cluster center from the generating point on the line")
parser.add_argument("--clusters", action="store", type=int,required=True,help="Total number of clusters constructed during runtime")
parser.add_argument("--points", action="store", type=int,required=True,help="Points per cluster.")
parser.add_argument("--workloadtype", action="store", choices=["insert","insert_delete"],required=True,help="Type of queries in the workload")
parser.add_argument("--history", action="store", type=int,required=True,help="Maximum number of clusters maintained additionally to the currently active one (insert+delete), number of clusters present before the workload queries are executed (insert,insert_delete)")
parser.add_argument("--steps", action="store", type=int,required=True,help="Steps along the line")
parser.add_argument("--dimensions", action="store", type=int,required=True,help="Number of dimensions")
parser.add_argument("--queriesperstep", action="store", type=int,required=True,help="Number of queries per step")
parser.add_argument("--maxprob", action="store", type=float,required=True,help="Maximum probability for the active cluster to be queried")


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

workloadtype = args.workloadtype

steps = args.steps
queries_per_step = args.queriesperstep

margin = args.margin
max_prob = args.maxprob

table = "%s_d%i" % (args.table,dimension)
query_output = args.queryoutput
data_output = args.dataoutput

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
#Do we want to have an index on the cluster for fast deletion?
tuples = []
for c in centers:    
    r = random.normal(0,sigma,(points*2,dimension)) #Create a few more points in case of uneven step counts
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

#Write $history clusters to data_output for loading tuples
for tuple_list in tuples[:history]:
    for point in tuple_list[:points]:
        writer.writerow((current,)+tuple(point))
    current += 1        
file.close()



#Set position vector to the point on line belonging to the last history cluster
pos_vector = pos_vector + (dir_vector/clusters)*history 


#Adjust direction vector accordingly
dir_vector = dir_vector * (1 - (float(history)/clusters))

tix = 0    
for i in range(1,steps+1):
    #Calculate the point on the line for the current step
    base = pos_vector + (dir_vector/steps) * i    
    
    #Move current cluster and delete the last one if necessary
    while(current < clusters and base[0] > centers[current][0]):
        current += 1
        if(workloadtype == "insert_delete"):
            output.write(delete_template % (current-history-1))
            output.write("VACUUM %s;\n" % table)
        tix = 0
        
    #Add a few tuples        
    for _ in range(0,(((clusters-history)*points/steps))):
        output.write(insert_template % ((current,)+tuple(tuples[current][tix])))
        tix += 1
                

        
    #Caclulate progress along the line normalized to the interval [0...1]    
    progress = (dir_vector[0]/(clusters-history) - (centers[current][0]-base[0]))/(dir_vector[0]/(clusters-history))  
    #Calculate probabilities for querying each cluster     
    if(workloadtype == "insert_delete"):
        probs = mc.calc_prob(current,progress,max_prob, clusters,current-history)
    else:
        probs = mc.calc_prob(current,progress,max_prob, clusters,0)
    
    #Pick target clusters for queries
    query_centers = random.choice(clusters, queries_per_step, p=probs)  

    #Generate queries
    queries = []
    for i in query_centers:
        c = centers[i] + random.normal(0, sigma, (dimension))
        rng = random.uniform(0, 3*sigma, dimension)

        low = c - rng 
        high = c + rng 
        
        output.write(query_template % tuple(mc.createBoundsList(low,high)))
        
output.close()      
