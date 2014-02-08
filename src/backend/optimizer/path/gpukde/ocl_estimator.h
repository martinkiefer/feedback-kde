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
	/* Buffers that keeps the current bandwidth estiamte*/
	cl_mem bandwidth_buffer;
	/* Fields for the sample and for sample maintenance */
	unsigned int rows_in_table;   // Current number of tuples in the table.
	unsigned int rows_in_sample;  // Current number of tuples in the sample.
	size_t sample_buffer_size;      // Size of the sample buffer in bytes.
	cl_mem sample_buffer;           // Buffer that stores the data sample.
	cl_mem sample_quality_buffer;   // Buffer that stores the quality factors for each sample item.
	/* Normalization factors */
	double* scale_factors;    // Scale factors that were applied to the data in the sample.
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
ocl_estimator_t* ocl_getEstimator(Relation rel);

// #########################################################################
// ################## FUNCTIONS FOR SAMPLE MANAGEMENT ######################

/*
 * sizeOfSampleItem
 *
 * Returns the size of a single sample item for the given estimator in bytes.
 */
size_t ocl_sizeOfSampleItem(ocl_estimator_t* estimator);

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
