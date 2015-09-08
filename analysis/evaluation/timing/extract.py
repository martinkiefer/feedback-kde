#! /usr/bin/env python

import argparse
import csv
import os
import sys

from collections import defaultdict
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument("--file", action="store", required=True)
parser.add_argument("--queries", action="store", type=int, required=True)
args = parser.parse_args()

# Delete all dat files in the current folder.
#for fn in os.listdir('.'):
#    if os.path.isfile(fn):
#        if ".dat" in fn:
#            os.remove(fn)

experiments = defaultdict(list) 

# Extract the plot files.
with open(args.file, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    reader.next()
    for row in reader:
      if not row: continue
      dim = row[0]
      model = row[1]
      gpu = row[2]
      if model == "stholes":
         modelsize = 2*int(row[3])
      else:
         modelsize = int(row[3])
         
         
      if gpu == "True":
         model = "%s_%s_gpu" % (dim, model)
      else:
         model = "%s_%s_cpu" % (dim, model)
      if "stholes" in model:
        experiments[model].append((row[3], int(row[5])))
      elif model <> "none":
        experiments[model].append((row[3], int(row[5]) + int(row[6])))

# Subtract the average runtime.
for model in experiments:
  f = open("%s.dat" % model, "w")
  for run in experiments[model]:
    f.write("%s\t%f\n" % (run[0], float(run[1]) / (1000 * args.queries)))
  f.close()
