#/bin/bash

# Postgres specific variables:
PSQL="psql"
PGDATABASE="mheimel"
PDATAFOLDER="/home/mheimel/postgres-kde/data"

# MonetDB specific variables:
MONETDATABASE=""

# Variables for the Genhist forest dataset:
COVTYPE_FILE=""	# Can be left empty (Script will download the data file instead.)

# Variables for the Genhist TPCH dataset:
DBGEN_FOLDER="/home/mheimel/tpch/dbgen" # Required.
