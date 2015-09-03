#! /usr/bin/env python

import argparse
import csv
import os
import sys

from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument("--file", action="store", required=True)
args = parser.parse_args()

# Delete all dat files in the current folder.
for fn in os.listdir('.'):
    if os.path.isfile(fn):
        if ".dat" in fn:
            os.remove(fn)

experiments = set()

# Now extract the plot files.
model = ""
error = 0
queries = 0
with open(args.file, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    next(reader)
    for row in reader:
      if not row: continue

      if model != row[3] or modelsize != row[4]:
          if model != "":
             f = open("%s.dat" % (model), "a")
             f.write("%s\t%s\n" % (modelsize, float(error)/queries))
             f.close()
             print "Model: %s, Sample Size %s, Error %s" % (model,modelsize,error)
          model = row[3]
          modelsize = row[4]
          error = 0
          queries = 0

      error += abs(int(row[5])-int(row[6]))
      queries += 1
