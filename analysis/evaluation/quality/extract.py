#! /usr/bin/env python

import argparse
import csv
import os
import sys
import re
from scipy.stats.mstats import mquantiles
from collections import defaultdict
from subprocess import call


def calculateWhiskers(values):
        #Calculate the median
        values.sort()
        lower_half = None
        upper_half = None

        q =  mquantiles(values)
        #Calculate the first and third quantile by taking the median
        first_quantile = q[0]
        third_quantile = q[2]
        inter_quartile_range = (third_quantile - first_quantile)

        ##Calculate the position of the whsikers
        lower_bound = first_quantile - 1.5 * inter_quartile_range
        upper_bound = third_quantile + 1.5 * inter_quartile_range
        for val in values:
                if lower_bound <= val:
                        lower_bound = val
                        break
        for val in reversed(values):
                if upper_bound >= val:
                        upper_bound = val
                        break
        return (lower_bound,upper_bound)

def calculateYrange(measurements):
        min_value = float('inf')
        max_value = -float('inf')
        for series in measurements:
                l,u = calculateWhiskers(series)
                min_value = min(min_value,l)
                max_value = max(max_value,u)
        #We have the minimum and maximum whiskers position in the plot
        #Now add a few percent slack.
        inter_value_range = max_value - min_value
        max_value += inter_value_range * 0.025 + 10e-9

        min_value -= inter_value_range * 0.025 + 10e-9
        return (min_value,max_value)


parser = argparse.ArgumentParser()
parser.add_argument("--file", action="store", required=True)
args = parser.parse_args()

# Delete all dat files in the current folder.
for fn in os.listdir('.'):
    if os.path.isfile(fn):
        if ".dat" in fn:
            os.remove(fn)

experiments = set()

dataset = ""
dim = ""
workload = ""

rows = {}
rows["forest"] = float(581012)
rows["bike"] = float(17379)
rows["power"] = float(2075259)
rows["protein"] = float(45730)
rows["synthetic"] = float(1000000)

dimensions = set()
workloads = set()
datasets = set()

# Fixed ranges to make the plots look nicer:

def createFileName(dataset, dim, workload, model):
  return "%s_%s_%s_%s.dat" % (dataset, dim, workload, model)

def writeExperiment(record):
  by_experiment = {}
  for dataset,workload,dim,model in record:
    dimensions.add(dim)
    workloads.add(workload)
    datasets.add(dataset)
    f = open("%s" % createFileName(dataset, dim, workload, model), "a")

    if (dataset,workload,dim) in by_experiment:
      by_experiment[(dataset,workload,dim)].append(record[(dataset,workload,dim,model)])
    else:
      by_experiment[(dataset,workload,dim)] = [ record[(dataset,workload,dim,model)] ]

    for e in record[(dataset,workload,dim,model)]:
      f.write("%e\n" % e)
    f.close()

  return by_experiment

ex = None

# Now extract the data.
with open(args.file, 'r') as csvfile:
  reader = csv.reader(csvfile, delimiter=';', quotechar="'")
  record = {}
  prev = None
  cur = None
  sum_error = 0.0
  first = True
  for row in reader:
    if not row: continue
    if row[2] != '_':
      m = re.match("([A-Za-z]*)",row[0])
      dataset = m.groups()[0]
      dataset = dataset.replace("normalized", "")
      dataset = dataset.replace("_","")
      dataset = dataset.replace(" ","")
      dataset = dataset.replace("set","synthetic")
      
      dim = row[1]
      experiments.add("%s_%s" % (workload, dim))
      workload = row[2]

    model = row[3]
    if first:
    	prev = (dataset,workload,dim,model)   
        first = False

    cur = (dataset,workload,dim,model) 
    if cmp(prev,cur) != 0:
      if prev in record:
        record[prev].append(sum_error/300.0)
      else: 
        record[prev] = [sum_error/300.0]
      sum_error = 0.0
      prev = cur

    # Extract the error for this experiment.
    error = abs(int(row[5])-int(row[6]))
    sum_error += float(error) / rows[dataset]

  record[cur].append(sum_error/300.0)
  ex = writeExperiment(record)

ranges = {}
for dataset, workload, dim in ex:
	ranges["%s_%s_%s" % (dataset, dim, workload)] = calculateYrange(ex[(dataset, workload, dim)])


# Generate a plot for all files:
sf = open("plot.sh", "w")

for d in dimensions:
  # Generate a new gnuplot file.
  gf = open("%s.gnuplot" % d, "w")
  gf.write("set terminal pdf size 17cm,10cm\n")
  gf.write("set style data boxplot\n")
  gf.write("set style boxplot labels off nooutliers sorted\n")
  gf.write("set style fill pattern 0.25\n")
  gf.write("set xrange [ -0.07 : 0.67 ]\n")
  gf.write("set lmargin 4.6\n")
  gf.write("set rmargin 0.1\n")
  gf.write("set tmargin 0.9\n")
  gf.write("set bmargin 0.6\n")
  gf.write("unset xtics\n")
  gf.write("set format y '%.1e'\n")
  #gf.write("set colorsequence podo\n")
  gf.write("set autoscale yfixmin\n")
  gf.write("set autoscale yfixmax\n")
  gf.write("set grid ytics\n")
  gf.write("set style fill transparent solid 0.5 border -1.0\n")
  gf.write("set ytics out offset 0.6 font ',7' nomirror\n")
  gf.write("set multiplot\n")
  xpos = 0
  ypos = 0.77
  for workload in sorted(workloads):
    for dataset in sorted(datasets):
      # Check whether we have fixed ranges for this plot.
      r = ()
      if ("%s_%s_%s" % (dataset, d, workload) in ranges):
        r = ranges["%s_%s_%s" % (dataset, d, workload)]
      if r:
        gf.write("set yrange [%e:%e]\n" % (r[0], r[1]))
      # Write the position.
      gf.write("set size 0.2,0.23\n")
      gf.write("set origin %f,%f\n" % (xpos, ypos))
      # And compute the new position.
      xpos += 0.2
      if (xpos == 1.0):
        ypos -= 0.23
        xpos = 0
      # Plot the title.
      gf.write("set title '%s (%s)' offset 0,-0.85 font 'Verdana Bold,10'\n" % (dataset, workload))
      # The lowest row should have tics for the models.
      if workload == "uv":
        gf.write("set xtics out nomirror offset 0,0.4 rotate by 40 right ('STHoles' 0, 'Heuristic' 0.15, 'SCV' 0.3, 'Batch' 0.45, 'Adaptive' 0.6) font 'Verdana Bold,9'\n")
        gf.write("set bmargin 0.2\n")
      else:
        gf.write("set xtics out nomirror ('' 0, '' 0.15, '' 0.3, '' 0.45, '' 0.6) \n")
      # Finally, plot the data.
      gf.write("plot '%s' using (0.0):1:(0.1) notitle, \\\n" % createFileName(dataset, d, workload, "stholes"))
      gf.write("'%s' using (0.15):1:(0.1) notitle, \\\n" % createFileName(dataset, d, workload, "kde_heuristic"))
      gf.write("'%s' using (0.3):1:(0.1) notitle,  \\\n" % createFileName(dataset, d, workload, "kde_scv"))
      gf.write("'%s' using (0.45):1:(0.1) notitle, \\\n" % createFileName(dataset, d, workload, "kde_batch"))
      gf.write("'%s' using (0.6):1:(0.1) notitle \n" % createFileName(dataset, d, workload, "kde_adaptive"))
      if r:
        gf.write("unset yrange\n")
  gf.write("unset multiplot\n")
  gf.close()
  sf.write("gnuplot %s.gnuplot > %s.pdf\n" % (d, d))
sf.close()
