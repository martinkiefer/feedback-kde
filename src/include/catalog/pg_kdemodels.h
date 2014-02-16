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
  int32   rowcount_table;
  int32   rowcount_sample;
  int32   sample_buffer_size;
#ifdef CATALOG_VARLEN
  float8  scale_factors[1];
  float4  bandwidth[1];
  text    sample_file;
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
#define Natts_pg_kdemodels                        8
#define Anum_pg_kdemodels_table                   1
#define Anum_pg_kdemodels_columns                 2
#define Anum_pg_kdemodels_rowcount_table          3
#define Anum_pg_kdemodels_rowcount_sample         4
#define Anum_pg_kdemodels_sample_buffer_size      5
#define Anum_pg_kdemodels_scale_factors           6
#define Anum_pg_kdemodels_bandwidth               7
#define Anum_pg_kdemodels_sample_file             8

#endif /* PG_KDEMODELS_H_ */
