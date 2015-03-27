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

// Forward declaration for sample and model maintenance data structures.
struct ocl_sample_optimization;
struct ocl_bandwidth_optimization;

typedef struct ocl_stats{
  long estimation_transfer_to_device;
  long estimation_transfer_to_host;
  
  long maintenance_transfer_to_device;
  long maintenance_transfer_to_host;
  long maintenance_transfer_time;
  
  long optimization_transfer_to_device;
  long optimization_transfer_to_host; 
  
  long nr_of_estimations;
  long nr_of_deletions;
  long nr_of_insertions;
} ocl_stats_t; 

/*
 * Definition of a constructed KDE estimator.
 */
typedef struct ocl_estimator {
  /* Information about the scope of this estimator */
  Oid table;    // For which table is this estimator configured?
  int32 columns;	 // Bitmap encoding which columns are stored in the estimator.
  unsigned int* column_order; // Order of the columns on the device.
  /* statistics about the estimator */
  unsigned int nr_of_dimensions;
  /* Buffers that keeps the current bandwidth.*/
  cl_mem bandwidth_buffer;
  /* Fields for the sample. */
  unsigned int rows_in_table;   // Current number of tuples in the table.
  unsigned int rows_in_sample;  // Current number of tuples in the sample.
  size_t sample_buffer_size;    // Size of the sample buffer in bytes.
  cl_mem sample_buffer;         // Buffer to store the data sample.
  /* Fields for the estimator. */
  cl_mem input_buffer;          // Buffer to store query bounds.
  cl_mem local_results_buffer;  // Buffer to store the local selectivities.
  cl_mem result_buffer;         // Buffer to store the final estimate.
  cl_kernel kde_kernel;         // Kernel to compute the estimate.
  ocl_aggregation_descriptor_t* sum_descriptor; // Descriptor for the final summation operation.
  /* Model optimization structures */
  struct ocl_bandwidth_optimization* bandwidth_optimization;
  struct ocl_sample_optimization* sample_optimization;
  struct ocl_stats* stats;
  /* Runtime information */
  bool open_estimation;     // Set to true if this estimator has produced a valid estimation for which we are still awaiting feedback.
  double last_selectivity;  // Stores the last selectivity computed by this estimator.
} ocl_estimator_t;

/*
 * SELECTOR FOR THE KERNEL TYPE.
 */
typedef enum ocl_kernel_type {
  GAUSS,
  EPANECHNIKOV
} ocl_kernel_type_t;

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
                            HeapTuple tuple, kde_float_t* target);

/*
 * pushEntryToSampleBuffer
 *
 * Pushes the given entry to the given position (in tuples) in the estimator
 * sample buffer.
 */
void ocl_pushEntryToSampleBufer(ocl_estimator_t* estimator, int position,
                                kde_float_t* data_item);

// Forward declaration for the bandwidth representation.
extern int kde_bandwidth_representation;

#endif /* USE_OPENCL */
#endif /* ESTIMATOR_H_ */
