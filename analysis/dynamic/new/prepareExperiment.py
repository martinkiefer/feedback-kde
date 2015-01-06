#! /usr/bin/env python

import argparse
import getpass
import math
import random
import re
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--queryfile1", action="store", required=True, help="First file with queries.")
parser.add_argument("--queryfile2", action="store", required=True, help="Second file with queries.")
parser.add_argument("--fraction", action="store", type=float, default=1.0, help="Fraction of used queries from first file.")
parser.add_argument("--train_queries", action="store", type=int, required=True, help="Number of queries in training set.")
parser.add_argument("--test_queries", action="store", type=int, required=True, help="Number of queries in test set.")
args = parser.parse_args()


# Count the number of queries in both sets.
f = open(args.queryfile1)
for total_queries_file1, _ in enumerate(f):
    pass
total_queries_file1 += 1
f.close()
f = open(args.queryfile2)
for total_queries_file2, _ in enumerate(f):
    pass
total_queries_file2 += 1
f.close()

# Determine how many queries we need to select from each file ...
queries_file1 = int(args.fraction * (args.train_queries + args.test_queries))
queries_file2 = (args.train_queries + args.test_queries) - queries_file1

# Now read the selected number of random queries into memory.
queries = []
if (queries_file1 > 0):
  selected_queries = range(1, total_queries_file1 + 1)
  random.shuffle(selected_queries)
  selected_queries = set(selected_queries[0:queries_file1])
  f = open(args.queryfile1)
  for linenr, line in enumerate(f):
    if linenr in selected_queries:
      queries.append(line)
  f.close()
if (queries_file2 > 0):
  selected_queries = range(1, total_queries_file2)
  random.shuffle(selected_queries)
  selected_queries = set(selected_queries[0:queries_file2])
  f = open(args.queryfile2)
  for linenr, line in enumerate(f):
    if linenr in selected_queries:
      queries.append(line)
  f.close()

# Ok, finally we shuffle and split the selected queries into the training
# and the test set.
random.shuffle(queries)
train_set = queries[:args.train_queries]
test_set = queries[args.train_queries:]

# Write the training set:
wf = open("/tmp/train_queries_%s.sql" % getpass.getuser(), "w")
for query in train_set:
  wf.write(query)
wf.close()

# And the test set:
wf = open("/tmp/test_queries_%s.sql" % getpass.getuser(), "w")
for query in test_set:
  wf.write(query)
wf.close()
