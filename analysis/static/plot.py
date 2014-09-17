from collections import namedtuple
from collections import OrderedDict
import csv
import math
import numpy
from os import system
from sets import Set
import sys

models = Set()
workloads = ["dv", "dt", "uv", "ut"]
experiments = Set()
averaged_measurements = OrderedDict()

def plotExperimentalSeries(experiment):
    # Open the gnuplot file
    f = open('/tmp/plot.dat', 'w')
    f.write("x\t")
    for model in models:
        f.write("\"%s\"\t" % model.replace("_", " "))
    f.write("\n")
    for workload in workloads:
        f.write("%s\t" % workload)
        for model in models:
            key = (experiment[0], experiment[1], workload, model)
            if key in averaged_measurements:
                f.write("%f\t" % averaged_measurements[key])
            else:
                f.write("0\t")
        f.write("\n")
    f.close()
    # Now generate the gnuplot script
    f = open('/tmp/plot.gnuplot', 'w')
    f.write("set terminal pdf enhanced\n")
    f.write("set output '%s/%s_%s.pdf'\n" % (sys.argv[2], experiment[0], experiment[1]))
    f.write("set style data histogram\n")
    f.write("set style fill solid border rgb 'black'\n")
    f.write("set auto x\n")
    f.write("set ylabel 'Median relative estimation error [%]'\n")
    f.write("set logscale y\n")
    f.write("set tmargin 2.2\n")
    f.write("set ylabel font ',15'\n")
    f.write("set xtics font ',17'\n")
    f.write("set key font ',15'\n")
    f.write("set key at graph 1.01,1.12 vertical maxrows 1 samplen 3 width -1 spacing 2\n")
    f.write("set grid y\n")
    f.write("set yrange [0.01:10000]\n")
    f.write("set xrange [-0.5:3.6]\n")
    f.write("plot '/tmp/plot.dat' using 4:xtic(1) title col fs pattern 2,")
    f.write("'' using 5:xtic(1) title col fs pattern 5,")
    f.write("'' using 2:xtic(1) title col fs pattern 6,")
    f.write("'' using 6:xtic(1) title col fs pattern 7\n") 
    f.close()
    # And call gnuplot
    system('gnuplot /tmp/plot.gnuplot')

# Average all experiments
with open(sys.argv[1], 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')    
    Experiment = namedtuple('experiment', ['dataset', 'dimensions', 'workload', 'selectivity', 'model', 'modelsize', 'trainqueries', 'errortype', 'error'])    
    for row in map(Experiment._make, csvreader):
        experiment_key = (row.dataset, row.dimensions, row.workload, row.model)
        models.add(row.model)
        experiments.add((row.dataset, row.dimensions))
        if math.isnan(float(row.error)):
            continue
        if experiment_key in averaged_measurements:
            error = averaged_measurements[experiment_key]
            error.append(float(row.error))
        else:
            averaged_measurements[experiment_key] = [float(row.error)]

for experiment in averaged_measurements:
    measurement = averaged_measurements[experiment]
    averaged_measurements[experiment] = numpy.median(numpy.array(measurement))

for experiment in experiments:
    plotExperimentalSeries(experiment)
