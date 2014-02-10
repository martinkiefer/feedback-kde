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
	Oid table;    // For which table ss this estimator configured?
	int32 columns;	 // Bitmap encoding which columns are stored in the estimator.
	AttrNumber* column_order; // Order of the columns on the device.
	/* Some statistics about the estimator */
	unsigned int nr_of_dimensions; 
	/* Buffers that keeps the current bandwidth estimate*/
	cl_mem bandwidth_buffer;
	/* Fields for the sample */
	unsigned int rows_in_table;   // Current number of tuples in the table.
	unsigned int rows_in_sample;  // Current number of tuples in the sample.
	size_t sample_buffer_size;      // Size of the sample buffer in bytes.
	cl_mem sample_buffer;           // Buffer that stores the data sample.
	/* These two buffers are used to track a linear regression between sample contributions and the expected results */
	cl_mem sample_quality_slopes_buffer;
	cl_mem sample_quality_intercepts_buffer;
	/* Normalization factors */
	double* scale_factors;    // Scale factors that were applied to the data in the sample.
	/* Runtime information */
	bool open_estimation;     // Set to true if this estimator has produced a valid estimation for which we are still waiting for feedback.
	double last_selectivity;  // Stores the last selectivity computed by this estimator.
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
 * Fetch an estimator for a relation (We allow one estimator per relation).
 */
ocl_estimator_t* ocl_getEstimator(Oid relation);

// #########################################################################
// ################## FUNCTIONS FOR SAMPLE MANAGEMENT ######################

/*
 * sizeOfSampleItem
 *
 * Returns the size of a single sample item for the given estimator in bytes.
 */
size_t ocl_sizeOfSampleItem(ocl_estimator_t* estimator);

/*
 * ocl_maxRowsInSample
 *
 * Returns the maximum number of rows that can possibly be stored in the
 * sample for the given estimator.
 */
unsigned int ocl_maxRowsInSample(ocl_estimator_t* estimator);

/*
 * extractSampleTuple
 *
 * Extracts the columns required by the estimator from the provided tuple
 * and writes them into the target buffer.
 */
void ocl_extractSampleTuple(ocl_estimator_t* estimator, Relation rel,
                            HeapTuple tuple, float* target);

/*
 * pushEntryToSampleBuffer
 *
 * Pushes the given entry to the given position (in tuples) in the estimator
 * sample buffer.
 */
void ocl_pushEntryToSampleBufer(ocl_estimator_t* estimator, int position,
                                float* data_item);

#endif /* USE_OPENCL */
#endif /* ESTIMATOR_H_ */
