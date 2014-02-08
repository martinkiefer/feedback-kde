#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
/*

 * ocl_estimator.c
 *
 *  Created on: 25.05.2012
 *      Author: mheimel
 */

#include "ocl_estimator.h"
#include "ocl_utilities.h"

#ifdef USE_OPENCL

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#include "miscadmin.h"
#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/sysattr.h"
#include "catalog/pg_kdemodels.h"
#include "catalog/pg_type.h"
#include "storage/lock.h"
#include "storage/ipc.h"
#include "utils/array.h"
#include "utils/fmgroids.h"
#include "utils/rel.h"
#include "utils/tqual.h"

extern bool ocl_use_gpu;
extern bool enable_kde_estimator;
extern int kde_samplesize;

/*
 * Global registry variable
 */
ocl_estimator_registry_t* registry = NULL;

static cl_kernel prepareKDEKernel(ocl_estimator_t* estimator) {
  // First, fetch the correct program.
  cl_kernel kernel = ocl_getKernel("epanechnikov_kde",
                                   estimator->nr_of_dimensions);
  // Now configure the kernels.
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &(estimator->sample_buffer));
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &(ocl_getContext()->result_buffer));
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &(ocl_getContext()->input_buffer));
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &(estimator->bandwidth_buffer));
  return kernel;
}

static float sumOfArray(cl_mem input_buffer, cl_mem result_buffer,
                        unsigned int elements) {
	ocl_context_t* context = ocl_getContext();
	// Fetch the required sum kernels:
	cl_kernel fast_sum = ocl_getKernel("sum_par", 0);
	cl_kernel slow_sum = ocl_getKernel("sum_seq", 0);
	clFinish(context->queue);
	struct timeval start; gettimeofday(&start, NULL);
	// Determine the kernel parameters:
	size_t local_size = context->max_workgroup_size;
	size_t processors = context->max_compute_units;
	// Figure out how many elements we can aggregate per thread in the parallel part:
	unsigned int tuples_per_thread = elements / (processors * local_size);
	// Now compute the configuration of the sequential kernel:
	unsigned int slow_kernel_data_offset = processors * tuples_per_thread * local_size;
	unsigned int slow_kernel_elements = elements - slow_kernel_data_offset;
	unsigned int slow_kernel_result_offset = processors;
	// Ok, we selected the correct kernel and parameters. Now prepare the arguments.
	cl_int err = 0;
	err |= clSetKernelArg(fast_sum, 0, sizeof(cl_mem), &input_buffer);
	err |= clSetKernelArg(fast_sum, 1, sizeof(cl_mem), &result_buffer);
	err |= clSetKernelArg(fast_sum, 2, sizeof(unsigned int), &tuples_per_thread);
	err |= clSetKernelArg(slow_sum, 0, sizeof(cl_mem), &input_buffer);
	err |= clSetKernelArg(slow_sum, 1, sizeof(unsigned int), &slow_kernel_data_offset);
	err |= clSetKernelArg(slow_sum, 2, sizeof(unsigned int), &slow_kernel_elements);
	err |= clSetKernelArg(slow_sum, 3, sizeof(cl_mem), &result_buffer);
	err |= clSetKernelArg(slow_sum, 4, sizeof(unsigned int), &slow_kernel_result_offset);
	// Fire the kernel
	if (tuples_per_thread) {
		size_t global_size = local_size * processors;
		err |= clEnqueueNDRangeKernel(context->queue, fast_sum, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	}
	if (slow_kernel_elements)
		err |= clEnqueueTask(context->queue, slow_sum, 0, NULL, NULL);
	clEnqueueBarrier(context->queue);
	// Now transfer the partial aggregates back to a local buffer ...
	float* partial_aggregates = (float*)malloc(sizeof(float)*(1+processors));
	err |= clEnqueueReadBuffer(context->queue, result_buffer, CL_TRUE, 0, (1+processors)*sizeof(float), partial_aggregates, 0, NULL, NULL);
	// .. and compute the final aggregate.
	float result = 0;
	if (tuples_per_thread) {
		result = partial_aggregates[0];
		unsigned int i;
		for (i = 1; i < processors; ++i)
			result += partial_aggregates[i];
	}
	if (slow_kernel_elements)
		result += partial_aggregates[processors];
	// Clean up ...
	clReleaseKernel(slow_sum);
	clReleaseKernel(fast_sum);
	free(partial_aggregates);
	// .. and return the result.
	struct timeval now; gettimeofday(&now, NULL); long seconds = now.tv_sec - start.tv_sec; long useconds = now.tv_usec - start.tv_usec;
	long mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	fprintf(stderr, "Sum: %li ms\n", mtime);
	return result;
}

static float rangeKDE(ocl_context_t* ctxt, ocl_estimator_t* estimator) {
  cl_kernel kde_kernel = prepareKDEKernel(estimator);
  size_t global_size = estimator->rows_in_sample;
	clEnqueueNDRangeKernel(ctxt->queue, kde_kernel, 1, NULL, &global_size,
	                       NULL, 0, NULL, NULL);
	clEnqueueBarrier(ctxt->queue);
	float result = sumOfArray(ctxt->result_buffer, ctxt->result_buffer,
	                          estimator->rows_in_sample);
	result /= (float)estimator->rows_in_sample;
	clReleaseKernel(kde_kernel);
	return result;
}

#define Natts_pg_kdemodels                        9
#define Anum_pg_kdemodels_table                   1
#define Anum_pg_kdemodels_columns                 2
#define Anum_pg_kdemodels_rowcount_table          3
#define Anum_pg_kdemodels_rowcount_sample         4
#define Anum_pg_kdemodels_sample_buffer_size      5
#define Anum_pg_kdemodels_scale_factors           6
#define Anum_pg_kdemodels_bandwidth               7
#define Anum_pg_kdemodels_sample                  8
#define Anum_pg_kdemodels_sample_quality          9


static ocl_estimator_t* ocl_buildEstimatorFromCatalogEntry(Relation kde_rel,
                                                           HeapTuple tuple) {
  unsigned int i;
  Datum datum;
  ArrayType* array;
  bytea* byte_array;
  bool isNull;
  ocl_context_t* context = ocl_getContext();
  ocl_estimator_t* descriptor = calloc(1, sizeof(ocl_estimator_t));

  // >> Read the table identifier.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_table,
                       RelationGetDescr(kde_rel), &isNull);
  descriptor->table = DatumGetObjectId(datum);

  // >> Read the dimensionality and the column order.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_columns,
                       RelationGetDescr(kde_rel), &isNull);
  descriptor->columns = DatumGetInt32(datum);
  unsigned int columns = descriptor->columns;
  for (i=0; columns && i<32; ++i) {
    if (columns & (0x1)) {
      descriptor->nr_of_dimensions++;
      descriptor->column_order = realloc(
          descriptor->column_order, descriptor->nr_of_dimensions * sizeof(AttrNumber));
      descriptor->column_order[descriptor->nr_of_dimensions - 1] = (AttrNumber)i;
    }
    columns >>= 1;
  }

  // >> Read the table rowcount.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_rowcount_table,
                         RelationGetDescr(kde_rel), &isNull);
  descriptor->rows_in_table = DatumGetInt32(datum);

  // >> Read the sample rowcount.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_rowcount_sample,
                         RelationGetDescr(kde_rel), &isNull);
  descriptor->rows_in_sample = DatumGetInt32(datum);

  // >> Read the sample buffer size.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_sample_buffer_size,
                         RelationGetDescr(kde_rel), &isNull);
  descriptor->sample_buffer_size = DatumGetInt32(datum);

  // >> Read the scale factors.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_scale_factors,
                       RelationGetDescr(kde_rel), &isNull);
  array = DatumGetArrayTypeP(datum);
  descriptor->scale_factors = malloc(
      sizeof(float8) * descriptor->nr_of_dimensions);
  memcpy(descriptor->scale_factors, (float8*)ARR_DATA_PTR(array),
         descriptor->nr_of_dimensions * sizeof(float8));

  // >> Read the bandwidth.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_bandwidth,
                       RelationGetDescr(kde_rel), &isNull);
  array = DatumGetArrayTypeP(datum);
  descriptor->bandwidth_buffer = clCreateBuffer(
        context->context, CL_MEM_READ_WRITE,
        sizeof(float)*descriptor->nr_of_dimensions, NULL, NULL);
  clEnqueueWriteBuffer(context->queue, descriptor->bandwidth_buffer, CL_FALSE,
                       0, sizeof(float)*descriptor->nr_of_dimensions,
                       (char*)ARR_DATA_PTR(array), 0, NULL, NULL);

  // >> Read the sample.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_sample,
                       RelationGetDescr(kde_rel), &isNull);
  byte_array = DatumGetByteaP(datum);
  descriptor->sample_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      descriptor->sample_buffer_size, NULL, NULL);
  clEnqueueWriteBuffer(
      context->queue, descriptor->sample_buffer, CL_FALSE, 0,
      ocl_sizeOfSampleItem(descriptor) * descriptor->rows_in_sample,
      VARDATA(byte_array), 0, NULL, NULL);

  // >> Read the sample quality.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_sample_quality,
                       RelationGetDescr(kde_rel), &isNull);
  byte_array = DatumGetByteaP(datum);
  descriptor->sample_quality_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      descriptor->sample_buffer_size, NULL, NULL);
  clEnqueueWriteBuffer(
      context->queue, descriptor->sample_quality_buffer, CL_FALSE, 0,
      sizeof(float) * descriptor->rows_in_sample,
      VARDATA(byte_array), 0, NULL, NULL);

  // Wait for all transfers to finish.
  clFinish(context->queue);
  // We are done.
  return descriptor;
}

static void ocl_updateEstimatorInCatalog(ocl_estimator_t* estimator) {
  unsigned int i;
  HeapTuple tuple;
  Datum* array_datums;
  ArrayType* array;
  ocl_context_t* context = ocl_getContext();

  Datum values[Natts_pg_kdemodels];
  bool  nulls[Natts_pg_kdemodels];
  bool  repl[Natts_pg_kdemodels];
  memset(values, 0, sizeof(values));
  memset(nulls, false, sizeof(nulls));
  memset(repl, true, sizeof(repl));

  // >> Write the table identifier.
  values[Anum_pg_kdemodels_table-1] = ObjectIdGetDatum(estimator->table);

  // >> Write the dimensionality and the column order.
  values[Anum_pg_kdemodels_columns-1] = Int32GetDatum(estimator->columns);

  // >> Write the table rowcount.
  values[Anum_pg_kdemodels_rowcount_table-1] = Int32GetDatum(
      estimator->rows_in_table);

  // >> Write the sample rowcount.
  values[Anum_pg_kdemodels_rowcount_sample-1] = Int32GetDatum(
      estimator->rows_in_sample);

  // >> Write the sample buffer size.
  values[Anum_pg_kdemodels_sample_buffer_size-1] = Int32GetDatum(
      (unsigned int)(estimator->sample_buffer_size));

  // >> Write the scale factors.
  array_datums = palloc(sizeof(Datum) * estimator->nr_of_dimensions);
  for (i = 0; i < estimator->nr_of_dimensions; ++i) {
    array_datums[i] = Float8GetDatum(estimator->scale_factors[i]);
  }
  array = construct_array(array_datums, estimator->nr_of_dimensions,
                          FLOAT8OID, sizeof(double), FLOAT8PASSBYVAL, 'i');
  values[Anum_pg_kdemodels_scale_factors-1] = PointerGetDatum(array);

  // >> Write the bandwidth.
  float* host_bandwidth = palloc(estimator->nr_of_dimensions * sizeof(float));
  clEnqueueReadBuffer(
      context->queue, estimator->bandwidth_buffer, CL_TRUE, 0,
      estimator->nr_of_dimensions * sizeof(float), host_bandwidth,
      0, NULL, NULL);
  for (i = 0; i < estimator->nr_of_dimensions; ++i) {
    array_datums[i] = Float4GetDatum(host_bandwidth[i]);
  }
  pfree(host_bandwidth);
  array = construct_array(array_datums, estimator->nr_of_dimensions,
                          FLOAT4OID, sizeof(float), FLOAT4PASSBYVAL, 'i');
  values[Anum_pg_kdemodels_bandwidth-1] = PointerGetDatum(array);

  // >> Write the sample.
  bytea* host_sample = palloc(
      ocl_sizeOfSampleItem(estimator) * estimator->rows_in_sample + VARHDRSZ);
  clEnqueueReadBuffer(
      context->queue, estimator->sample_buffer, CL_FALSE, 0,
      ocl_sizeOfSampleItem(estimator) * estimator->rows_in_sample,
      VARDATA(host_sample), 0, NULL, NULL);
  SET_VARSIZE(
      host_sample,
      ocl_sizeOfSampleItem(estimator) * estimator->rows_in_sample + VARHDRSZ);
  values[Anum_pg_kdemodels_sample-1] = PointerGetDatum(host_sample);

  // >> Write the sample quality.
  bytea* host_sample_quality = palloc(
      sizeof(float) * estimator->rows_in_sample + VARHDRSZ);
  clEnqueueReadBuffer(
      context->queue, estimator->sample_quality_buffer, CL_FALSE, 0,
      ocl_sizeOfSampleItem(estimator) * estimator->rows_in_sample,
      VARDATA(host_sample_quality), 0, NULL, NULL);
  SET_VARSIZE(host_sample_quality,
              sizeof(float) * estimator->rows_in_sample + VARHDRSZ);
  values[Anum_pg_kdemodels_sample_quality-1] = PointerGetDatum(host_sample_quality);

  // Wait for the sample transfers to finish.
  clFinish(context->queue);

  // Ok, we constructed the tuple. Now try to find whether the estimator is
  // already present in the catalogue.
  Relation kdeRel = heap_open(KdeModelRelationID, RowExclusiveLock);
  ScanKeyData key[1];
  ScanKeyInit(&key[0], Anum_pg_kdemodels_table, BTEqualStrategyNumber, F_OIDEQ,
              ObjectIdGetDatum(estimator->table));
  HeapScanDesc scan = heap_beginscan(kdeRel, SnapshotNow, 1, key);
  tuple = heap_getnext(scan, ForwardScanDirection);
  if (!HeapTupleIsValid(tuple)) {
    // This is a new estimator. Insert it into the table.
    heap_endscan(scan);
    tuple = heap_form_tuple(RelationGetDescr(kdeRel),
                            values, nulls);
    simple_heap_insert(kdeRel, tuple);
  } else {
    // This is an existing estimator. Update the corresponding tuple.
    heap_endscan(scan);
    HeapTuple newtuple = heap_modify_tuple(tuple, RelationGetDescr(kdeRel),
                                           values, nulls, repl);
    simple_heap_update(kdeRel, &tuple->t_self, newtuple);
  }
  heap_close(kdeRel, RowExclusiveLock);

  // Clean up.
  pfree(array_datums);
  pfree(host_sample);
  pfree(host_sample_quality);
}

/*
 *  Static helper function to release the resources held by a single estimator.
 *
 *  If materialize is set, the function will write the changes to stable
 *  storage.
 */
static void ocl_freeEstimator(ocl_estimator_t* estimator, bool materialize) {
  // Write all changes to stable storage.
  if (materialize) ocl_updateEstimatorInCatalog(estimator);
  // Now release the in-memory structure.
  if (estimator->scale_factors)
    free(estimator->scale_factors);
  if (estimator->sample_buffer)
    clReleaseMemObject(estimator->sample_buffer);
  if (estimator->sample_quality_buffer)
    clReleaseMemObject(estimator->sample_quality_buffer);
  if (estimator->bandwidth_buffer)
    clReleaseMemObject(estimator->bandwidth_buffer);
}

static void ocl_releaseRegistry() {
  if (!registry) return;
  unsigned int i;
  // Update all registered estimators within the system catalogue.
  for (i=0; i<registry->estimator_directory->entries; ++i) {
    ocl_estimator_t* estimator = (ocl_estimator_t*)directory_valueAt(
        registry->estimator_directory, i);
    ocl_freeEstimator(estimator, true);
  }
  // Now release the registry.
  directory_release(registry->estimator_directory, true);
  free(registry);
  registry = NULL;
}

static void
ocl_cleanUpRegistry(int code, Datum arg) {
  fprintf(stderr, "Cleaning up OpenCL and materializing KDE models.\n");
  // Open a new transaction to ensure that we can write back any changes.
  AbortOutOfAnyTransaction();
  StartTransactionCommand();

  ocl_releaseRegistry();

  CommitTransactionCommand();
}

/*
 * Initialize the registry.
 */
static void ocl_initializeRegistry() {
	if (registry) return; // Don't reintialize.
  if (IsBootstrapProcessingMode()) return; // Don't initialize during bootstrap.
  if (ocl_getContext() == NULL) return; // Don't run if OpenCl is not initialized.

	// Allocate a new descriptor.
	registry = calloc(1, sizeof(ocl_estimator_registry_t));
	registry->estimator_bitmap = calloc(1, 4096); // Enough bits for 32768 tables.
	registry->estimator_directory = directory_init(sizeof(Oid), 20);

	// Now open the KDE estimator table, read in all stored estimators and
	// register their descriptors.
	Relation kdeRel = heap_open(KdeModelRelationID, RowExclusiveLock);
	HeapScanDesc scan = heap_beginscan(kdeRel, SnapshotNow, 0, NULL);
	HeapTuple tuple;
	while ((tuple = heap_getnext(scan, ForwardScanDirection)) != NULL) {  // Compute the normalization factor.
	  ocl_estimator_t* estimator = ocl_buildEstimatorFromCatalogEntry(
	      kdeRel, tuple);
	  if (estimator == NULL) continue;
	  // Note: Right now, we only support a single estimator per table.
	  directory_insert(registry->estimator_directory, &(estimator->table),
	                   estimator);
	  registry->estimator_bitmap[estimator->table / 8]
	                             |= (0x1 << estimator->table % 8);
	}
	heap_endscan(scan);
	heap_close(kdeRel, RowExclusiveLock);
	// Finally, register a cleanup function to ensure we write any estimator
	// changes back to the catalogue.
	on_shmem_exit(ocl_cleanUpRegistry, 0);
}

// Helper function to fetch (and initialize if it does not exist) the registry
static ocl_estimator_registry_t* ocl_getRegistry(void) {
  if (!registry) ocl_initializeRegistry();
  return registry;
}


/* Custom comparator for a range request */
static int compareRange(const void* a, const void* b) {
	if ( *(AttrNumber*)a > *(AttrNumber*)b )
		return 1;
	else if ( *(AttrNumber*)a == *(AttrNumber*)b )
		return 0;
	else
		return -1;
}

/*
 * Dump the request to stderr.
 */
void ocl_dumpRequest(ocl_estimator_request_t* request) {
	unsigned int i;
	if (!request)
		return;
	fprintf(stderr, "Firing request for table: %i:\n", request->table_identifier);
	for (i=0; i<request->range_count; ++i)
		fprintf(stderr, "\tColumn %i in: [%f , %f]\n", request->ranges[i].colno, request->ranges[i].lower_bound, request->ranges[i].upper_bound);
}

int ocl_updateRequest(ocl_estimator_request_t* request,
		AttrNumber colno, float* lower_bound, float* upper_bound) {
	/*
	 * First, make sure to find the range entry for the given column.
	 * If no column exists, insert a new one.
	 */
	ocl_colrange_t* column_range = NULL;
	if (request->ranges == NULL) {
		/* We have added no ranges so far. Add this range as the first entry. */
		request->ranges = (ocl_colrange_t*)malloc(sizeof(ocl_colrange_t));
		request->range_count++;
		request->ranges->colno = colno;
		request->ranges->lower_bound = -FLT_MAX;
		request->ranges->upper_bound = FLT_MAX;
		column_range = request->ranges;
	} else {
		/* Check whether we already have a range for this column */
		column_range = bsearch(
				&colno, request->ranges, request->range_count,
				sizeof(ocl_colrange_t), &compareRange);
		if (column_range == NULL) {
			/* We have to add the column. Add storage for a new value */
			request->range_count++;
			request->ranges = (ocl_colrange_t*)realloc(request->ranges, sizeof(ocl_colrange_t)*(request->range_count));
			/* Initialize the new column range */
			column_range = &(request->ranges[request->range_count-1]);
			column_range->colno = colno;
			column_range->lower_bound = -FLT_MAX;
			column_range->upper_bound = FLT_MAX;
			/* Now we have to re-sort the array */
			qsort(request->ranges, request->range_count, sizeof(ocl_colrange_t), &compareRange);
			/* Ok, we inserted the value. Use bsearch again to get the final position of our newly inserted range. */
			column_range = bsearch(
							&colno, request->ranges, request->range_count,
							sizeof(ocl_colrange_t), &compareRange);
		}
	}
	/* Now update the found range entry with the new information */
	if (lower_bound) {
		if (column_range->lower_bound <= *lower_bound) {
			column_range->lower_bound = *lower_bound;
		}
	}
	if (upper_bound) {
		if (column_range->upper_bound >= *upper_bound) {
			column_range->upper_bound = *upper_bound;
		}
	}
	return 1;
}

int ocl_estimateSelectivity(const ocl_estimator_request_t* request,
		Selectivity* selectivity) {
	struct timeval start; gettimeofday(&start, NULL); 
	unsigned int i,j;
	// Fetch the OpenCL context, initializing it if requested.
	ocl_context_t* ctxt = ocl_getContext();
	if (ctxt == NULL)	return 0;
	// Make sure that the registry is initialized
	if (registry == NULL)	ocl_initializeRegistry();
	// Check the registry, whether we have an estimator for the requested table.
	if (!(registry->estimator_bitmap[request->table_identifier / 8]
	                                 & (0x1 << request->table_identifier % 8)))
	  return 0;
	ocl_estimator_t* estimator = DIRECTORY_FETCH(
	    registry->estimator_directory, &(request->table_identifier),
	    ocl_estimator_t);
	if (estimator == NULL) return 0;
	// Check if the request can potentially be answered by the estimator:
	if (request->range_count > estimator->nr_of_dimensions) return 0;
	// Now check if all columns in the request are covered by the estimator:
	int request_columns = 0;
	for (i = 0; i < request->range_count; ++i) {
	  request_columns |= 0x1 << request->ranges[i].colno;
	}
	if ((estimator->columns | request_columns) != estimator->columns) return 0;
	// Cool, prepare a request to the estimator
	float* row_ranges = (float*)malloc(2*sizeof(float)*estimator->nr_of_dimensions);
	for (i = 0; i < estimator->nr_of_dimensions; ++i) {
		row_ranges[2*i] = -FLT_MAX;
		row_ranges[2*i+1] = FLT_MAX;
	}
	j = 0;
	int found = 1;
	for (i = 0; i < request->range_count; ++i) {
		found = 0;
		for (;!found && j < estimator->nr_of_dimensions; ++j) {
			if (estimator->column_order[j] == request->ranges[i].colno) {
				// Make sure we adjust the request to the re-scaled data.
				row_ranges[2*j] = request->ranges[i].lower_bound / estimator->scale_factors[j];
				row_ranges[2*j+1] = request->ranges[i].upper_bound / estimator->scale_factors[j];
				found = 1;
			}
		}
		if (!found) break;
	}
	if (found) {
		// Prepare the requst
		clEnqueueWriteBuffer(ctxt->queue, ctxt->input_buffer, CL_TRUE, 0,
               2*estimator->nr_of_dimensions*sizeof(float), row_ranges, 
               0, NULL, NULL);
	  *selectivity = rangeKDE(ctxt, estimator);
		// Print timing: 
		struct timeval now; gettimeofday(&now, NULL); 
		long seconds = now.tv_sec - start.tv_sec; 
		long useconds = now.tv_usec - start.tv_usec;
		long mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5; 
		fprintf(stderr, "Estimated selectivity: %f, took: %ld ms.\n", *selectivity, mtime);
		free(row_ranges);
		return 1;
	} else {
		free(row_ranges);
		return 0;
	}
}

unsigned int ocl_maxSampleSize(unsigned int dimensionality) {
	return (kde_samplesize*1024)/(dimensionality*sizeof(float));
}

void ocl_constructEstimator(
    Relation rel, unsigned int rows_in_table, unsigned int dimensionality,
    AttrNumber* attributes, unsigned int sample_size, HeapTuple* sample) {
	unsigned int i, j;
	// Make sure we have a context
	ocl_context_t* ctxt = ocl_getContext();
	if (ctxt == NULL)	return;
	// Make sure the registry exists.
	if (!registry) ocl_initializeRegistry();
	// Some Debug output
	fprintf(stderr, "Constructing an estimator for table %i.\n", rel->rd_node.relNode);
	fprintf(stderr, "\tColumns:");
	for (i=0; i<dimensionality; ++i)
		fprintf(stderr, " %i", attributes[i]);
	fprintf(stderr, "\n");
	fprintf(stderr, "\tUsing a backing sample of %i out of %i tuples.\n", sample_size, rows_in_table);
	// Insert a new estimator.
	ocl_estimator_t* estimator = calloc(1, sizeof(ocl_estimator_t));
	ocl_estimator_t* old_estimator = directory_insert(
	    registry->estimator_directory, &(rel->rd_node.relNode), estimator);
	// If there was an existing estimator, release it.
	if (old_estimator) ocl_freeEstimator(estimator, false);
	// Update the bitmap.
  registry->estimator_bitmap[rel->rd_node.relNode / 8]
                             |= (0x1 << rel->rd_node.relNode % 8);
	// Update the descriptor info.
	estimator->table = rel->rd_node.relNode;
  estimator->nr_of_dimensions = dimensionality;
  estimator->column_order = calloc(1, dimensionality * sizeof(AttrNumber));
	for (i = 0; i<dimensionality; ++i) {
	  estimator->columns |= 0x1 << attributes[i];
	  estimator->column_order[i] = attributes[i];
	}
	estimator->rows_in_sample = sample_size;
	estimator->rows_in_table = rows_in_table;
	/* 
    * Ok, we set up the estimator. Prepare the sample for shipping it to the device. 
    * While preparing the sample, we compute the variance for each column on the fly.
 	 * The variance is then used to normalize the data to unit variance in each dimension.
    */
	float* host_buffer = (float*)malloc(
	    ocl_sizeOfSampleItem(estimator) * sample_size);
	double* mean = (double*)calloc(1, sizeof(double) * dimensionality);
	double* M2 = (double*)calloc(1, sizeof(double) * dimensionality);
	for ( i = 0; i < sample_size; ++i) {
	  // Extract the item.
	  ocl_extractSampleTuple(estimator, rel, sample[i],
	                         &(host_buffer[i*ocl_sizeOfSampleItem(estimator)]));
	  // And update the attributes.
	  for ( j = 0; j < estimator->nr_of_dimensions; ++j) {
	    float delta = host_buffer[i*ocl_sizeOfSampleItem(estimator) + j] - mean[j];
	    mean[j] = mean[j] + delta / (i + 1);
	    M2[j] += delta * (host_buffer[i*ocl_sizeOfSampleItem(estimator) + j] - mean[j]);
	  }
	}
	// Compute the scale factors
	free(mean);
	estimator->scale_factors = M2;
	for ( i = 0; i < estimator->nr_of_dimensions; ++i) {
	  estimator->scale_factors[i] /= (sample_size - 1);
	  estimator->scale_factors[i] = sqrt(estimator->scale_factors[i]);
	}
	// Re-scale the data to unit variance.
	for ( j = 0; j < sample_size; ++j) {
		for ( i = 0; i < dimensionality; ++i) {
			host_buffer[j*dimensionality + i] /= estimator->scale_factors[i];
		}
	}
	// Compute an initial bandwidth estimate using Scott's rule.
	float* bandwidth = (float*)malloc(sizeof(float)*dimensionality);
	for ( i = 0; i < dimensionality; ++i) {
		bandwidth[i] = pow(sample_size, -1.0f/(float)(dimensionality+4))*sqrt(5);
	}
	estimator->bandwidth_buffer = clCreateBuffer(
	    ctxt->context, CL_MEM_READ_WRITE, sizeof(float) * dimensionality,
	    NULL, NULL);
  clEnqueueWriteBuffer(
      ctxt->queue, estimator->bandwidth_buffer, CL_TRUE, 0,
      sizeof(float) * dimensionality, bandwidth, 0, NULL, NULL);
	// Print some debug info.
	fprintf(stderr, "\tInitial bandwidth guess:");
	for ( i = 0; i< dimensionality ; ++i)
		fprintf(stderr, " %f", bandwidth[i]);
	fprintf(stderr, "\n");
	free(bandwidth);
	// Push the sample to the device.
	estimator->sample_buffer_size = kde_samplesize * 1024;
	estimator->sample_buffer = clCreateBuffer(
	    ctxt->context, CL_MEM_READ_WRITE, estimator->sample_buffer_size,
	    NULL, NULL);
	clEnqueueWriteBuffer(
	    ctxt->queue, estimator->sample_buffer, CL_TRUE, 0,
	    sample_size * ocl_sizeOfSampleItem(estimator), host_buffer,
	    0, NULL, NULL);
  free(host_buffer);
  // Prepare the sample quality buffer.
  estimator->sample_quality_buffer = clCreateBuffer(
      ctxt->context, CL_MEM_READ_WRITE,
      estimator->sample_buffer_size, NULL, NULL);
  // And fill it with zeros.
  cl_kernel init_zero = ocl_getKernel("init_zero", 0);
  size_t global_size = estimator->sample_buffer_size / sizeof(float);
  clEnqueueNDRangeKernel(ocl_getContext()->queue, init_zero, 1, NULL,
                         &global_size, NULL, 0, NULL, NULL);
  clFinish(ocl_getContext()->queue);
}

void assign_ocl_use_gpu(bool newval, void *extra) {
	if (newval != ocl_use_gpu) {
		ocl_releaseRegistry();
		ocl_releaseContext();
	}
}

void assign_kde_samplesize(int newval, void *extra) {
  if (newval != kde_samplesize) {
    ocl_releaseRegistry();
    ocl_releaseContext();
    // TODO: Clear all estimators that have a larger sample size than this.
  }
}

void assign_enable_kde_estimator(bool newval, void *extra) {
  if (newval != enable_kde_estimator) {
    ocl_releaseRegistry();
    ocl_releaseContext();
  }
}

bool ocl_useKDE(void) {
  return enable_kde_estimator;
}

ocl_estimator_t* ocl_getEstimator(Relation rel) {
  ocl_estimator_registry_t* registry = ocl_getRegistry();
  Oid relation_id = rel->rd_id;
  // Check the bitmap whether we have an estimator for this relation.
  if (!(registry->estimator_bitmap[relation_id / 8]
          & (0x1 << (relation_id % 8))))
      return NULL;
  return DIRECTORY_FETCH(registry->estimator_directory, &relation_id,
                         ocl_estimator_t);
}

size_t ocl_sizeOfSampleItem(ocl_estimator_t* estimator) {
  return estimator->nr_of_dimensions * sizeof(float);
}

void ocl_pushEntryToSampleBufer(ocl_estimator_t* estimator, int position,
                                float* data_item) {
  ocl_context_t* context = ocl_getContext();
  size_t transfer_size = ocl_sizeOfSampleItem(estimator);
  size_t offset = position * transfer_size;
  clEnqueueWriteBuffer(context->queue, estimator->sample_buffer, CL_FALSE,
                       offset, transfer_size, data_item, 0, NULL, NULL);
  // Also zero out the sample quality buffer.
  float zero = 0;
  clEnqueueWriteBuffer(context->queue, estimator->sample_quality_buffer,
                       CL_FALSE, position * sizeof(float), sizeof(float),
                       &zero, 0, NULL, NULL);
  clFinish(context->queue);
}

void ocl_extractSampleTuple(ocl_estimator_t* estimator, Relation rel,
                            HeapTuple tuple, float* target) {
  unsigned int i;
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    unsigned int colNr = estimator->column_order[i];
    Oid attribute_type = rel->rd_att->attrs[colNr-1]->atttypid;
    bool isNull;
    if (attribute_type == FLOAT4OID)
      target[i] = DatumGetFloat4(heap_getattr(tuple, colNr, rel->rd_att, &isNull));
    else if (attribute_type == FLOAT8OID)
      target[i] = DatumGetFloat8(heap_getattr(tuple, colNr, rel->rd_att, &isNull));
  }
}

#endif /* USE_OPENCL */
