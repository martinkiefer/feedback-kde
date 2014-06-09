# -*- coding: utf-8 -*-

from numpy import random
from numpy import add
import csv
#import matplotlib.pyplot as plt

#plt.show

#Target file
target = "data_gen1_d"

#Number of points
points = 10**6

#Share of points that are picked independent of cluster centers
error = 0.1

#Number of clusters
clusters = 100

#Dimensions
dimensions = [2,3,4,5,8,10]

#Maximum standard deviation
sigma = 0.03




def generate_clusters(clusters,points_per_cluster, dimensions, sigma):
    points = []
    
    centers = random.random_sample((clusters,dimensions))
    for center in centers:
        for delta in random.normal(0,sigma,(points_per_cluster,dimensions)):
            points.append(add(center,delta))
    return points


def generate_noise(points, dimensions):
    return random.random_sample((points,dimensions))


def create_dimension(file_prefix, dimensions,points,clusters,sigma,error):
    
    points_per_cluster = points * (1- error) / clusters
    file = "%s%i%s" % (file_prefix,dimensions,".csv")
    result = open(file,'wb')
    writer = csv.writer(result)
    
    print("Creating "+file)
    print("Generating clusters for dimensionality %i" % dimensions)
    cluster_points = generate_clusters(clusters,points_per_cluster, dimensions, sigma)
    writer.writerows(cluster_points)
    print("Done.")
    #for point in cluster_points:
        #plt.plot(point[0],point[1], 'ro')

    print("Generating noise for dimensionality %i" % dimensions)
    noise_points = generate_noise(points*error,dimensions)
    writer.writerows(noise_points)
    print("Done")
    #for point in noise_points:
        #plt.plot(point[0],point[1], 'ro')
        
    result.close()
    print("Finished %s" % file)
    #plt.show()
        

#And here weg go.
for i in dimensions:
    create_dimension(target,i,points,clusters,sigma,error)

