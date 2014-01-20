/*
 * estimator.h
 *
 *  Created on: 25.05.2012
 *      Author: mheimel
 */

#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

#include "ocl_estimator_api.h"

#ifdef USE_OPENCL

/*
 * Definition of a constructed KDE estimator.
 */
typedef struct ocl_estimator {
	/* Information about the scope of this estimator */
	Oid relation_id;			/* For which relation is this estimator configured? */
	AttrNumber* column_numbers;	/* Which columns are stored within this estimator? */
	/* Some statistics about the estimator */
	unsigned int nr_of_dimensions; 
	size_t sample_size;
	/* Prepared kernels for the estimator */
	bool exact; // Do we use KDE or exact evaluation?
	cl_kernel range_kernel;
	/* Prepared buffers for the estimator */
	cl_mem bandwidth_buffer;
	cl_mem data_sample;
	/* Normalization factor */
	float* scale_factors; /* Normalization factors that were used to scale the data to unit variance */
	float range_normalization_factor;
} ocl_estimator_t;

/*
 * Performance characteristics.
 */
typedef struct ocl_perf_stats {
	float ms_per_tuple;
} ocl_perf_stats_t;

/*
 * Registry of all known estimators.
 */
typedef struct ocl_estimator_registry {
	/* List of estimators */
	unsigned int nr_of_estimators;
	ocl_estimator_t* estimators;
	/* Performance characteristics */
	ocl_perf_stats_t perf_stats;
} ocl_estimator_registry_t;

/*
 * Construct a new estimator descriptor:
 */
extern int ocl_prepareEstimator(ocl_estimator_t* estimator,
		unsigned int n, unsigned int d, bool full_sample,
		cl_mem buffer, float* bandwidth);

#endif /* USE_OPENCL */
#endif /* ESTIMATOR_H_ */
