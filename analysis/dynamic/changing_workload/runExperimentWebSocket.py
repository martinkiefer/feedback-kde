import argparse
import inspect
import os
import psycopg2
import sys
import json
import subprocess
import time
import datetime
import sys
import math

#from twisted.python import log
from twisted.internet import reactor
from autobahn.twisted.websocket import WebSocketServerProtocol
from autobahn.twisted.websocket import WebSocketServerFactory

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")#
parser.add_argument("--error", action="store", choices=["relative","absolute"], default="relative", help="Which error metric should be optimized / reported?")#
parser.add_argument("--optimization", action="store", choices=["heuristic", "adaptive", "stholes"], default="heuristic", help="How should the model be optimized?")#
parser.add_argument("--log", action="store", required=True, help="Where to append the experimental results?")#
args = parser.parse_args()


# Fetch the arguments.
dbname = args.dbname
errortype = args.error
optimization = args.optimization
log = args.log

class MyServerProtocol(WebSocketServerProtocol):
       
        # Extract the error from the error file.
   def extractError(self):
       #time.sleep(1)
       row = self.ifile.readline()
       row = row.split(" ; ")
       local_error = float(row[self.selected_col])*float(row[self.tuple_col])
       self.cur.execute("SELECT kde_get_stats('%s')" % self.table)
       tup = self.cur.fetchone()
       stats=tup[0][1:-1].split(",")
       data = {"error": local_error, "transfers": int(stats[7])+int(stats[8]), "time" : int(stats[9])}
       self.dump_file.write("%s\n" % json.dumps(data))
       self.sendMessage(json.dumps(data), False)

   def init(self,payload):
        basepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.conn = psycopg2.connect("dbname=%s host=localhost" % dbname)
        self.conf = json.loads(payload)
        self.finished_queries = 0
        print "bash %s" % os.path.join(basepath, "mvtc_id/load-mvtc_id-tables.sh")
        if "mvtc_id" in self.conf["workload"]:
	    spread = self.conf["workload"][8:]
	    ppc = self.conf["ppc"]
	    pqr = str(int((10*int(ppc))/(float(self.conf["pqr"])*100)))
	    
	    print "%s %s %s" % (spread,ppc,pqr)
            self.conn.set_session(autocommit=True)
            self.querypath = os.path.join(basepath, "mvtc_id/%s_%s_%s/queries" % (spread,ppc,pqr))
            self.table = "mvtc_id_d%s" % self.conf["dimensions"]
            subprocess.call(["bash", "%s" % os.path.join(basepath, "mvtc_id/load-mvtc_id-tables.sh"),spread,ppc,pqr])
            self.queryfile = "%s.sql" % (self.table)
        if "bike" in self.conf["workload"]:
	    self.conn.set_session('read uncommitted', autocommit=True)
            self.querypath = os.path.join(basepath, "../static/bike/queries")
            self.queryfile = "bike%sdt.sql" % self.conf["dimensions"]
            self.table = "bike%s" % self.conf["dimensions"]

        self.cur = self.conn.cursor()
	df = "%s_%s" % (self.table,self.conf["maintenance"])
	df = "%s_%s" % (df,self.conf["samplesize"])
        if(self.conf["maintenance"] == "TKR"):
            self.cur.execute("SET kde_sample_maintenance TO TKR;")
            self.cur.execute("SET kde_sample_maintenance_karma_threshold TO %s;" % self.conf["threshold"])	
            self.cur.execute("SET kde_sample_maintenance_karma_limit TO %s;" % self.conf["decay"])
	    df = "%s_%s" % (df,self.conf["threshold"])
	    df = "%s_%s" % (df,self.conf["decay"])
        if(self.conf["maintenance"] == "PKR"):
            self.cur.execute("SET kde_sample_maintenance TO PKR;")
            self.cur.execute("SET kde_sample_maintenance_period  TO %s;" % self.conf["period"] )
            self.cur.execute("SET kde_sample_maintenance_karma_limit TO %s;" % self.conf["decay"])
	    df = "%s_%s" % (df,self.conf["period"])
	    df = "%s_%s" % (df,self.conf["decay"])
        if(self.conf["maintenance"] == "CAR"):
            self.cur.execute("SET kde_sample_maintenance TO CAR;")
        if(self.conf["maintenance"] == "None"):
            self.cur.execute("SET kde_sample_maintenance TO NONE;")
        if(self.conf["maintenance"] == "PRR"):
            self.cur.execute("SET kde_sample_maintenance TO PRR;")
            self.cur.execute("SET kde_sample_maintenance_period  TO %s;" % self.conf["period"] )  
	    df = "%s_%s" % (df,self.conf["period"])
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')    
	self.dump_file = open("/tmp/%s_%s" % (df,st), "w")
        
	self.f = open(os.path.join(self.querypath, self.queryfile), "r")
        self.queries = len(self.f.readlines())
        self.f.seek(0)  
        
        # Set all required options.
        self.cur.execute("SET ocl_use_gpu TO true;")
        self.cur.execute("SET kde_estimation_quality_logfile TO '/tmp/error.log';")
        if (errortype == "relative"):
            self.cur.execute("SET kde_error_metric TO SquaredRelative;")
        elif (errortype == "absolute"):
            self.cur.execute("SET kde_error_metric TO Quadratic;")
        # Set the optimization strategy.
        if (optimization == "adaptive"):
            self.cur.execute("SET kde_enable TO true;")
            self.cur.execute("SET kde_enable_adaptive_bandwidth TO true;")
            self.cur.execute("SET kde_minibatch_size TO 5;")
            self.cur.execute("SET kde_samplesize TO %s;" % self.conf["samplesize"])
        elif (optimization == "heuristic"):
            self.cur.execute("SET kde_enable TO true;")
            self.cur.execute("SET kde_samplesize TO %s;" % self.conf["samplesize"])
        self.cur.execute("SET kde_debug TO false;")
	self.cur.execute("SELECT pg_backend_pid()")
	print self.cur.fetchone()
	sys.stdout.flush()
	#time.sleep(20)
        
        print "Building estimator ...",
        sys.stdout.flush()
        analyze_query = "ANALYZE %s(" % self.table
        for i in range(1, int(self.conf["dimensions"]) + 1):
            if (i>1):
                analyze_query += ", c%i" % i
            else:
                analyze_query += "c%i" %i
        analyze_query += ");"
        self.cur.execute(analyze_query)
        print "done!"
        
        print "Running experiment:"
        self.ifile  = open("/tmp/error.log", "rb")
        row = self.ifile.readline()
        row = row.split(" ; ")
        self.selected_col = -1
        self.tuple_col = -1
        column = 0
        for col in row:
            if (col.strip().lower() == errortype):
                self.selected_col = column
            if (col.strip().lower() == "tuples"):
                self.tuple_col = column
            column = column + 1
        if (self.selected_col == -1 or self.tuple_col == -1):
            print "Error-type %s or absolute tuple value not present in given file!" % errortype
            sys.exit()        
        
   def onMessage(self, payload, isBinary):
        if payload != "N":
            #This is the configuration message. Initialize.
            self.init(payload)
        
        while(True):
            line = self.f.readline()
            if(line == ""):
                self.sendMessage("D", False)
                return 
            
            #print line
            self.finished_queries += 1
            sys.stdout.write("\r\tFinished %i of %i queries." % (self.finished_queries, self.queries))
            sys.stdout.flush()
            try:
                self.cur.execute(line)
                if line == "":
                    break
                if "SELECT" == line[0:6]:
                    self.extractError()
                    return
                else:
                    continue

            except psycopg2.DatabaseError:
              print "Database error occured. Terminating."
              reactor.callFromThread(reactor.stop)
              
   def onClose(self,wasClean, code, reason):
        #reactor.callFromThread(reactor.stop)
        self.cur.close()
        self.dump_file.close()
        self.f.close()
        self.conn.close()
        self.ifile.close()
        
              
#from twisted.python import log as l
#l.startLogging(sys.stdout)
print "Get factpr"
factory = WebSocketServerFactory("ws://localhost:9000",debug = False)
print "Get protocol"
factory.protocol = MyServerProtocol
print "Listen"
reactor.listenTCP(9000, factory)
print "Run"
reactor.run()
print "Bye"
