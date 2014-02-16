/*
 * ocl_estimator_api.h
 *
 *  Created on: 06.06.2012
 *      Author: mheimel
 */

#ifndef OCL_ESTIMATOR_API_H_
#define OCL_ESTIMATOR_API_H_

#include "postgres.h"

#include "access/attnum.h"
#include "access/htup.h"
#include "nodes/nodes.h"
#include "utils/rel.h"
#include "utils/relcache.h"

#ifdef USE_OPENCL

/*
 * Structure defining a single range on a single column.
 */
typedef struct ocl_colrange {
	AttrNumber colno;
	double lower_bound;
	bool lower_included;
	double upper_bound;
	bool upper_included;
} ocl_colrange_t;

/*
 * Structure that captures a selectivity request for a given table and a number of
 * column ranges.
 */
typedef struct ocl_estimator_request {
	Oid table_identifier;
	unsigned int range_count;
	ocl_colrange_t* ranges;
} ocl_estimator_request_t;

/*
 * Enum definition to select a possible error metric that should be optimized.
 */
typedef enum error_metrics {
  ABSOLUTE = 0,
  RELATIVE = 1,
  QUADRATIC = 2,
  Q = 3
} ocl_error_metrics_t;

/*
 * Enum definition to select a batch optimziation strategy.
 */
typedef enum {
  CONSTRAINED,
  UNCONSTRAINED_PENALIZED
} kde_optimization_strategy_t;

/*
 * Function for updating a range request with new bounds on a given attribute.
 */
extern int ocl_updateRequest(ocl_estimator_request_t* request, AttrNumber column,
		double* lower_bound, bool lower_included, double* upper_bound, bool upper_included);

/*
 * Main entry function for the opencl selectivity estimator.
 */
int ocl_estimateSelectivity(const ocl_estimator_request_t* estimation_request, Selectivity* selectivity);

/*
 * Helper function to get the maximum sample size for KDE estimators.
 */
unsigned int ocl_maxSampleSize(unsigned int dimensionality);

/*
 * Functions to report estimation errors to a file.
 */
bool ocl_reportErrors();
void ocl_reportErrorToLogFile(Oid relation, double actual, double expected);

/*
 * Entry function for generating a KDE estimator
 */
void ocl_constructEstimator(Relation rel, unsigned int rows_in_table,
                            unsigned int dimensionality, AttrNumber* attributes,
                            unsigned int sample_size, HeapTuple* sample);

/*
 * Returns whether KDE should be used or not.
 */
bool ocl_useKDE(void);

/*
 * Helper functions for GUC that handle assignments for the configuration variables.
 */
extern void assign_ocl_use_gpu(bool newval, void *extra);
extern void assign_kde_enable(bool newval, void *extra);
extern void assign_kde_samplesize(int newval, void *extra);
extern void assign_kde_estimation_quality_logfile_name(const char *newval, void *extra);


/*
 * Functions for propagating informations to the estimator sample maintenanec..
 */
extern void ocl_notifySampleMaintenanceOfInsertion(Relation rel, HeapTuple new_tuple);
extern void ocl_notifySampleMaintenanceOfDeletion(Relation rel);

/*
 * Propagate selectivity information to the model maintenance.
 */
extern void ocl_notifyModelMaintenanceOfSelectivity(Oid rel, double selectivity);

#endif /* USE_OPENCL */
#endif /* OCL_ESTIMATOR_API_H_ */
