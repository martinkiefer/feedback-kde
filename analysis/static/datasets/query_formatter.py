#!/usr/bin/env python
import argparse
import csv
import sys

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--queries", action="store", required=True, help="CSV File containing the input data.")
parser.add_argument("--table", action="store", required=True, help="Name of the query table.")
parser.add_argument("--output", action="store", required=True, help="Name of the output sql file.")
args = parser.parse_args()

fout = open(args.output, "w")
with open(args.queries, "r") as fin:
   reader = csv.reader(fin, delimiter='|')
   template = None
   for line in reader:
      del line[-1]   # Remove the selectivity
      if not template:
         columns = len(line) / 2
         template = "SELECT count(*) FROM %s WHERE " % args.table
         for i in range(0, columns):
            if i>0:
               template += " AND "
            template += "c%i>{} AND c%i<{}" % (i+1, i+1)
         template += ";\n"
      fout.write(template.format(*line))
fout.close()
