import argparse                                                                 
import psycopg2
import re

def minQ(col_id, table):
   return "(SELECT min(c%i) FROM %s)" % (col_id, table) 

def maxQ(col_id, table):
   return "(SELECT max(c%i) FROM %s)" % (col_id, table) 

def avgQ(col_id, table):
   return "(SELECT avg(c%i) FROM %s)" % (col_id, table) 

def stdQ(col_id, table):
   return "(SELECT stddev(c%i) FROM %s)" % (col_id, table) 

# Define the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--dbname", action="store", required=True, help="Database to which the script will connect.")
parser.add_argument("--port", action="store", type=int, default=5432, help="Port of the postmaster.")
args = parser.parse_args()                                                      

# Open a connection to Postgres.
conn = psycopg2.connect("dbname=%s host=localhost port=%i" % (args.dbname, args.port))
conn.set_session('read uncommitted', autocommit=True)
cur = conn.cursor()

# Identify all tables.
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';");
tables = []
for table in cur:
   if "rescaled" in table[0]:
      continue
   if "normalized" in table[0]:
      continue
   m = re.match(".+([0-9]+)", table[0])
   if m:
      tables.append(table[0])
# Alright, now create both the normalized and the rescaled table.
for table in tables:
   m = re.match("(.+[^0-9])([0-9]+)", table)
   tableprefix = m.group(1)
   tabledim = int(m.group(2))
   # Construct the new table names.
   table_scaled = "%srescaled%i" % (tableprefix, tabledim)
   table_normal = "%snormalized%i" % (tableprefix, tabledim)
   
   # Drop potential existing tables:
   try:
      cur.execute("DROP table %s;" % table_scaled)   
   except:  # Ignore if the table does not exist.
      pass
   try:
      cur.execute("DROP table %s;" % table_normal)
   except:  # Ignore if the table does not exist.
      pass

   # Build the query to fill the rescaled table.
   query_scaled = "SELECT"
   for i in range(1, tabledim+1):
      if (i > 1):
         query_scaled += ", "
      query_scaled += "(c%i - %s)/(%s - %s) AS c%i" % (i, minQ(i, table), maxQ(i, table), minQ(i, table), i)
   query_scaled += " INTO %s FROM %s;" % (table_scaled, table)
   cur.execute(query_scaled)
   
   # Build the query to fill the normalized table.
   query_normal = "SELECT"
   for i in range(1, tabledim+1):
      if (i > 1):
         query_normal += ", "
      query_normal += "(c%i - %s) / %s AS c%i" % (i, avgQ(i, table), stdQ(i, table), i)
   query_normal += " INTO %s FROM %s;" % (table_normal, table)
   cur.execute(query_normal)

# Clean up.
cur.close()
conn.close()
