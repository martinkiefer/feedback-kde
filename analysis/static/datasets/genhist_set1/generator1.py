#!/usr/bin/env python

import argparse
import csv
import math
import numpy
from numpy import random
from numpy import add
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--points", action="store", type=int, default=1000000, help="Total number of points in the dataset.")
parser.add_argument("--dim", action="store", type=int, required=True, help="Dimensionality of the dataset.")
parser.add_argument("--noise", action="store", type=float, default=0.1, help="Fraction of noise in the datasaet.")
parser.add_argument("--clusters", action="store", type=int, default=100, help="Total number of clusters in the dataset.")
parser.add_argument("--outfile", action="store", required=True, help="Name of the output file.")
args = parser.parse_args()

# Figure out how our points are distributed.
points_in_clusters = int((1.0 - args.noise) * args.points)
points_per_cluster = points_in_clusters / args.clusters
points_in_noise = args.points - args.clusters * points_per_cluster

# Now generate the clusters.
points = []
min_cluster_width = 0#0.02
max_cluster_width = 0.8
created_clusters = 0
while (created_clusters < args.clusters):
   p1 = numpy.random.uniform(size = args.dim)
   p2 = numpy.random.uniform(size = args.dim)
   lower_left = numpy.minimum(p1, p2)
   upper_right = numpy.maximum(p1, p2)
   cluster_range = numpy.clip(upper_right - lower_left, min_cluster_width, max_cluster_width)
   if numpy.any(lower_left + cluster_range > 1):
      continue
   for i in range(0, points_per_cluster):
      points.append(lower_left + numpy.random.uniform(size=args.dim) * cluster_range)
   created_clusters += 1

# And generate the random noise.
for i in range(0, points_in_noise):
   points.append(numpy.random.uniform(size=args.dim))

# Shuffle the array a bit.
random.shuffle(points)

# And print the data points.
with open(args.outfile, "w") as fout:
   writer = csv.writer(fout, delimiter='|')
   writer.writerows(points)
sys.exit(-1)
