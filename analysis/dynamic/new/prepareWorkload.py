#! /usr/bin/env python

import argparse
import getpass
import math
import random
import re
import sys

def createRandomMixture(queries_1, queries_2, mixture, queries):
   queries_from_2 = int(mixture * queries)
   queries_from_1 = queries - queries_from_2
   queries = []
   # Generate a new mixture.
   if queries_from_1 > 0:
      selected_queries = range(0, len(queries_1))
      random.shuffle(selected_queries)
      selected_queries = selected_queries[0:queries_from_1]
      queries.extend([queries_1[i] for i in selected_queries])
   if queries_from_2 > 0:
      selected_queries = range(0, len(queries_2))
      random.shuffle(selected_queries)
      selected_queries = selected_queries[0:queries_from_2]
      queries.extend([queries_2[i] for i in selected_queries])
   random.shuffle(queries)
   return queries

parser = argparse.ArgumentParser()
parser.add_argument("--queryfile1", action="store", required=True, help="First file with queries.")
parser.add_argument("--queryfile2", action="store", required=True, help="Second file with queries.")
parser.add_argument("--mixture", action="store", required=True, help="Comma seperated mixture fractions in the workload.")
parser.add_argument("--queries_per_mixture", action="store", type=int, required=True, help="Number of queries per mixture.")
parser.add_argument("--trainqueries", default=0, type=int)
args = parser.parse_args()

# Read both queryfiles into memory.
queries_1 = []
f = open(args.queryfile1)
for query in f:
  queries_1.append(query)
f.close()
queries_2 = []
f = open(args.queryfile2)
for query in f:
  queries_2.append(query)
f.close()

# Extract the mixture configuration.
mixture = []
for fraction in args.mixture.split(','):
   mixture.append(float(fraction))

# Write the training set:
if args.trainqueries > 0:
  train_set = createRandomMixture(queries_1, queries_2, mixture[0], args.trainqueries)
  wf = open("/tmp/train_queries_%s.sql" % getpass.getuser(), "w")
  for query in train_set:
    wf.write(query)
  wf.close()

# And the test set:
wf = open("/tmp/test_queries_%s.sql" % getpass.getuser(), "w")
for fraction in mixture:
  queries = createRandomMixture(queries_1, queries_2, fraction, args.queries_per_mixture)
  for query in queries:
    wf.write(query)
wf.close()
