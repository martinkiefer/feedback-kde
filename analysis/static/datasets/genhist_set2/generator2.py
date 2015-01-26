# -*- coding: utf-8 -*-

import csv
from numpy import random
from numpy import add
import sys

#Target file
target = "gen2_d"

#Number of points
points = 10**6

#Share of points that are picked independent of cluster centers
error = 0.1

#Number of clusters
clusters = 50

#Dimensions
dimensions = [2,3,5,8]

#Maximum standard deviation
sigma = 0.03

def generate_clusters(clusters,points_per_cluster, dimensions, sigma):
    points = []
    
    centers = random.random_sample((clusters,dimensions))
    for center in centers:
        reduced_dimensions = random.random_integers(0,dimensions-1,2)
        
        for delta in random.normal(0,sigma,(points_per_cluster,dimensions)):
            point = add(center,delta)
            for dim in reduced_dimensions:
                point[dim]=random.random(); 
            points.append(point)
    return points


def generate_noise(points, dimensions):
    return random.random_sample((points,dimensions))


def create_dimension(directory, file_prefix, dimensions,points,clusters,sigma,error):
    
    points_per_cluster = points * (1- error) / clusters
    file = "%s/%s%i%s" % (directory, file_prefix, dimensions, ".csv")
    result = open(file,'wb')
    writer = csv.writer(result, delimiter='|')
    
    print("Creating "+file)
    print("Generating clusters for dimensionality %i" % dimensions)
    cluster_points = generate_clusters(clusters,points_per_cluster, dimensions, sigma)
    writer.writerows(cluster_points)
    print("Done.")

    print("Generating noise for dimensionality %i" % dimensions)
    noise_points = generate_noise(points*error,dimensions)
    writer.writerows(noise_points)
    print("Done")
        
    result.close()
    print("Finished %s%i" % (file_prefix, dimensions))       
        

#And here weg go.
for i in dimensions:
    create_dimension(sys.argv[1], target, i, points, clusters, sigma, error)

