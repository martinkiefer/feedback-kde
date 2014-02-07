/*
 * estimator.h
 *
 *  Created on: 25.05.2012
 *      Author: mheimel
 */

#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

#include "container/directory.h"
#include "optimizer/path/gpukde/ocl_estimator_api.h"

#include "ocl_utilities.h"

#ifdef USE_OPENCL

/*
 * Definition of a constructed KDE estimator.
 */
typedef struct ocl_estimator {
	/* Information about the scope of this estimator */
	Oid relation_id;			/* For which relation is this estimator configured? */
	int32 columns;	/* Bitmap encoding which columns are stored in this estimator */
	AttrNumber* column_order; /* Required order of the columns when sending requests. */
	/* Some statistics about the estimator */
	unsigned int nr_of_dimensions; 
	unsigned int sample_size;
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
 * Registry of all known estimators.
 */
typedef struct ocl_estimator_registry {
  // This encodes in a bitmap for which oids we have estimators.
  char* estimator_bitmap;
  // This stores an OID->estimator mapping.
	directory_t estimator_directory;
} ocl_estimator_registry_t;

/*
 * Initialize the kernels for a given estimator.
 */
extern int ocl_prepareEstimator(ocl_estimator_t* estimator);

#endif /* USE_OPENCL */
#endif /* ESTIMATOR_H_ */
