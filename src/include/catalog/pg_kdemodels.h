/*
 *
 * pg_kdemodels.h
 *    definition of system catalogue tables for the KDE models.
 *
 * src/include/catalog/pg_kdemodels.h
 *
 * NOTES
 *    the genbki.pl script reads this file and generates .bki
 *    information from the DATA() statements.
 *
 *-------------------------------------------------------------------------
 */

#ifndef PG_KDEMODELS_H_
#define PG_KDEMODELS_H_

#include "catalog/genbki.h"

/*
 * Definition of the KDE model table.
 */
#define KdeModelRelationID  3780

CATALOG(pg_kdemodels,3780) BKI_WITHOUT_OIDS
{
  Oid     table;
  int32   columns;
  bool    is_exact;
  int32   sample_size;
#ifdef CATALOG_VARLEN
  float4  scale_factors[1];
  float4  bandwidths[1];
  bytea   sample;
#endif
} FormData_pg_kdemodels;

/* ----------------
 *    Form_pg_kdemodels corresponds to a pointer to a tuple with
 *    the format of pg_kdemodels relation.
 * ----------------
 */
typedef FormData_pg_kdemodels *Form_pg_kdemodels;

/* ----------------
 *    compiler constants for pg_kdemodels
 * ----------------
 */
#define Natts_pg_kdemodels              7
#define Anum_pg_kdemodels_table         1
#define Anum_pg_kdemodels_columns       2
#define Anum_pg_kdemodels_is_exact      3
#define Anum_pg_kdemodels_sample_size   4
#define Anum_pg_kdemodels_scalefactors  5
#define Anum_pg_kdemodels_bandwidth     6
#define Anum_pg_kdemodels_sample        7

#endif /* PG_KDEMODELS_H_ */
