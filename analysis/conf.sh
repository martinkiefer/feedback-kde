#/bin/bash

# Postgres specific variables:
PSQL="/usr/local/pgsql/bin/psql"
PGDATABASE="xy"

# MonetDB specific variables:
MONETDATABASE=""

# Variables for the Genhist forest dataset:
COVTYPE_FILE=""	# Can be left empty (Script will download the data file instead.)

# Variables for the Genhist TPCH dataset:
DBGEN_FOLDER="/home/martin/Dokumente/HiWi/Data_Generation/dbgen" # Required.
