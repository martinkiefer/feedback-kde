#!/usr/bin/python
import psycopg2
import argparse
import itertools
import sys

# Define and parse the command line arguments:
parser = argparse.ArgumentParser()
# Database connectivity arguments.
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--port", action="store", type=int, default=5432, help="Port of the postmaster.")

args = parser.parse_args()

print "Establishing DB connection..."
try:
    conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (args.dbname, args.port))
    conn.set_session('read uncommitted', autocommit=True)
    cur = conn.cursor()
except Exception as e:
    print str(e)
    print "An error occurred while trying to connect to the database."
    sys.exit(-1)
print "Done"

tables = ["forest","set1_","power","protein","time","bike"]
dims = [3,8]

print "Checking for tables..."
try:
    for tab,dim in itertools.product(tables,dims):
        cur.execute("SELECT * FROM %s%s LIMIT 1;" % (tab,dim))
except Exception as e:
    print str(e)
    print "An error occurred while trying to check if all tables are present. Was CompileInstall.sh not executed properly?"
    sys.exit(-1)
print "Done"


print "Creating a KDE model on the CPU..."
try:
    for tab,dim in itertools.product(tables,dims):
        cur.execute("SET kde_debug TO false;") 
        cur.execute("SET ocl_use_gpu TO false;") 
        cur.execute("SET kde_error_metric TO Quadratic;") 
        cur.execute("DELETE FROM pg_kdefeedback;")
        cur.execute("ANALYZE bike3(c1,c2,c3);")
except Exception as e:
    print str(e)
    print "An error occured while trying to create a KDE model on the CPU. Is your OpenCL setup okay?"
    sys.exit(-1)
print "Done"


print "Creating a KDE model on the GPU..."
try:
    for tab,dim in itertools.product(tables,dims):
        cur.execute("SET kde_debug TO false;") 
        cur.execute("SET ocl_use_gpu TO true;") 
        cur.execute("SET kde_error_metric TO Quadratic;") 
        cur.execute("DELETE FROM pg_kdefeedback;")
        cur.execute("ANALYZE power3(c1,c2,c3);")
except Exception as e:
    print str(e)
    print "An error occured while trying to create a KDE model on the GPU. Is you OpenCL setup okay?"
    sys.exit(-1)
print "Done"


tables = ["forest","set1_","power","protein","bike"]
dims = [3,8]
workloads = ["ut","uv","dt","dv"]
print "Checking for query files..."
try:
    for tab,dim,wl in itertools.product(tables,dims,workloads):
       if tab == "set1_":
           fold = "genhist_set1"
       else:
           fold = tab
           file = './analysis/static/%s/queries/%s%s_%s_0.01.sql' % (fold,tab,dim,wl)
           num_lines = sum(1 for line in open(file))
           if num_lines != 2500:
               raise Exception("Queryfile %s does not have enough queries. (%s/%s)" % (file,num_lines,2500))    
except Exception as e:
        print str(e)
        print "An error occurred while trying to check if all query files were generated. Was the query generation process in CompileInstall.sh interrupted?"
        sys.exit(-1)
print "Done"

print "Post-install check passed."
