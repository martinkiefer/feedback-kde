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

#from twisted.python import log
from twisted.internet import reactor
from autobahn.twisted.websocket import WebSocketServerProtocol
from autobahn.twisted.websocket import WebSocketServerFactory

# Define and parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")#
parser.add_argument("--dataset", action="store", choices=["mvt", "mvtc_i","mvtc_id"], required=True, help="Which dataset should be run?")#
parser.add_argument("--error", action="store", choices=["relative","absolute"], default="relative", help="Which error metric should be optimized / reported?")#
parser.add_argument("--optimization", action="store", choices=["heuristic", "adaptive", "stholes"], default="heuristic", help="How should the model be optimized?")#
parser.add_argument("--log", action="store", required=True, help="Where to append the experimental results?")#
args = parser.parse_args()


# Fetch the arguments.
dbname = args.dbname
dataset = args.dataset
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
        self.cur = self.conn.cursor()
        self.conf = json.loads(payload)
        self.finished_queries = 0
        print "bash %s" % os.path.join(basepath, "mvtc_id/load-mvtc_id-tables.sh")
        if dataset == "mvt":
            self.conn.set_session('read uncommitted', autocommit=True)
            self.querypath = os.path.join(basepath, "mvt/queries")
            self.table = "mvtc_d%s" % self.conf["dimensions"]
            subprocess.call(["bash", "%s" % os.path.join(basepath, "mvt/load-mvt-tables.sh")])
        if dataset == "mvtc_i":
            self.conn.set_session(autocommit=True)
            self.querypath = os.path.join(basepath, "mvtc_i/queries")
            self.table = "mvtc_i_d%s" % self.conf["dimensions"]
            subprocess.call(["bash", "%s" % os.path.join(basepath, "mvtc_i/load-mvtc_i-tables.sh")] )
        if dataset == "mvtc_id":
            self.conn.set_session(autocommit=True)
            self.querypath = os.path.join(basepath, "mvtc_id/queries")
            self.table = "mvtc_id_d%s" % self.conf["dimensions"]
            subprocess.call(["bash", "%s" % os.path.join(basepath, "mvtc_id/load-mvtc_id-tables.sh")])
        self.queryfile = "%s.sql" % (self.table)

	df = "%s_%s" % (self.table,self.conf["maintenance"])
	df = "%s_%s" % (df,self.conf["samplesize"])
        if(self.conf["maintenance"] == "TKR"):
            self.cur.execute("SET kde_sample_maintenance TO TKR;")
            self.cur.execute("SET kde_sample_maintenance_karma_threshold TO %s;" % self.conf["threshold"])	
            self.cur.execute("SET kde_sample_maintenance_karma_decay TO %s;" % self.conf["decay"])
	    df = "%s_%s" % (df,self.conf["threshold"])
	    df = "%s_%s" % (df,self.conf["decay"])
        if(self.conf["maintenance"] == "TKRP"):
            self.cur.execute("SET kde_sample_maintenance TO TKRP;")
            self.cur.execute("SET kde_sample_maintenance_karma_threshold TO %s;" % self.conf["threshold"])
            self.cur.execute("SET kde_sample_maintenance_karma_decay TO %s;" % self.conf["decay"])
            self.cur.execute("SET kde_sample_maintenance_impact_decay TO %s;" % self.conf["impact_decay"])
	    df = "%s_%s" % (df,self.conf["threshold"])
	    df = "%s_%s" % (df,self.conf["decay"])
	    df = "%s_%s" % (df,self.conf["impact_decay"])
        if(self.conf["maintenance"] == "PKR"):
            self.cur.execute("SET kde_sample_maintenance TO PKR;")
            self.cur.execute("SET kde_sample_maintenance_period  TO %s;" % self.conf["period"] )
            self.cur.execute("SET kde_sample_maintenance_karma_decay TO %s;" % self.conf["decay"])
	    df = "%s_%s" % (df,self.conf["period"])
	    df = "%s_%s" % (df,self.conf["decay"])
        if(self.conf["maintenance"] == "CAR"):
            self.cur.execute("SET kde_sample_maintenance TO CAR;")
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
            self.cur.execute("SET kde_samplesize TO %i;" % self.conf["samplesize"])
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
                    #return
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
