/*
 * estimator.c
 *
 *  Created on: 25.05.2012
 *      Author: mheimel
 */

#include "ocl_estimator.h"
#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
#ifdef USE_OPENCL

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#include "miscadmin.h"
#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/sysattr.h"
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

// Definitions for the KDE estimator table.
#define KDE_TABLE_OID 3780
#define kde_table_num_rels 7
#define Anum_kde_table_relid 1
#define Anum_kde_table_columns 2
#define Anum_kde_table_is_exact 3
#define Anum_kde_table_sample_size 4
#define Anum_kde_table_scalefactors 5
#define Anum_kde_table_bandwidth 6
#define Anum_kde_table_sample 7

// Struct describing the KDE estimator catalogue table.
typedef struct FormData_kde_table {
  Oid   table;       /* For which table was this estimator created. */
  int32 columns;     /* Bitmap encoding which columns are contained in the estimator */
  int32 sample_size;
  bytea sample;
  bytea bandwidth;
} FormData_kde_table;

/*
 * Global registry variable
 */
ocl_estimator_registry_t* registry = NULL;

static float sumOfArray(cl_mem input_buffer, cl_mem result_buffer, unsigned int elements) {
	ocl_context_t* context = ocl_getContext();
	// Fetch the required sum kernels:
	cl_kernel fast_sum = ocl_getKernel("sum_par", "-DD=1 -DSAMPLE_SIZE=1");
	cl_kernel slow_sum = ocl_getKernel("sum_seq", "-DD=1 -DSAMPLE_SIZE=1");
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
  size_t global_size = estimator->sample_size;
	clEnqueueNDRangeKernel(ctxt->queue, estimator->range_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
	clEnqueueBarrier(ctxt->queue);
	return sumOfArray(ctxt->result_buffer, ctxt->result_buffer, estimator->sample_size) * estimator->range_normalization_factor;
}

static ocl_estimator_t* ocl_buildEstimatorFromCatalogEntry(Relation kde_rel,
                                                           HeapTuple tuple) {
  Datum datum;
  ArrayType* array;
  bool isNull;
  ocl_context_t* context = ocl_getContext();
  ocl_estimator_t* descriptor = calloc(1, sizeof(ocl_estimator_t));
  // Extract the table identifier.
  datum = heap_getattr(tuple, Anum_kde_table_relid,
                       RelationGetDescr(kde_rel), &isNull);
  descriptor->relation_id = DatumGetObjectId(datum);
  // Extract the dimensionality and the column order.
  datum = heap_getattr(tuple, Anum_kde_table_columns,
                       RelationGetDescr(kde_rel), &isNull);
  descriptor->columns = DatumGetInt32(datum);
  unsigned int i;
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
  // Extract the scale factors.
  datum = heap_getattr(tuple, Anum_kde_table_scalefactors,
                       RelationGetDescr(kde_rel), &isNull);
  array = DatumGetArrayTypeP(datum);
  descriptor->scale_factors = malloc(sizeof(float) * descriptor->nr_of_dimensions);
  memcpy(descriptor->scale_factors, (float*)ARR_DATA_PTR(array),
         descriptor->nr_of_dimensions * sizeof(float));
  // Extract whether the estimator is exact.
  datum = heap_getattr(tuple, Anum_kde_table_is_exact,
                       RelationGetDescr(kde_rel), &isNull);
  descriptor->exact = DatumGetBool(datum);
  if (!descriptor->exact) {
    // This is no exact estimator, extract the bandwidth.
    datum = heap_getattr(tuple, Anum_kde_table_bandwidth,
                         RelationGetDescr(kde_rel), &isNull);
    array = DatumGetArrayTypeP(datum);
    descriptor->bandwidth_buffer = clCreateBuffer(
        context->context, CL_MEM_READ_WRITE,
        sizeof(float)*descriptor->nr_of_dimensions, NULL, NULL);
    clEnqueueWriteBuffer(context->queue, descriptor->bandwidth_buffer, CL_TRUE, 0,
                         sizeof(float)*descriptor->nr_of_dimensions,
                         (char*)ARR_DATA_PTR(array), 0, NULL, NULL);
  }
  // Extract the sample size.
  datum = heap_getattr(tuple, Anum_kde_table_sample_size,
                       RelationGetDescr(kde_rel), &isNull);
  descriptor->sample_size = DatumGetInt32(datum);
  // Finally, extract the sample.
  datum = heap_getattr(tuple, Anum_kde_table_sample,
                          RelationGetDescr(kde_rel), &isNull);
  bytea* host_copy = DatumGetByteaP(datum);
  descriptor->data_sample = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(cl_float)*descriptor->sample_size*descriptor->nr_of_dimensions,
      NULL, NULL);
  clEnqueueWriteBuffer(
      context->queue, descriptor->data_sample, CL_TRUE, 0,
      sizeof(cl_float)*descriptor->sample_size*descriptor->nr_of_dimensions,
      VARDATA(host_copy), 0, NULL, NULL);
  // Now initialize the kernels.
  ocl_prepareEstimator(descriptor);
  // We are done.
  return descriptor;
}

static void ocl_updateEstimatorInCatalog(ocl_estimator_t* estimator) {
  unsigned int i;
  cl_int err = 0;
  HeapTuple tuple;
  Datum* array_datums;
  ArrayType* array;
  ocl_context_t* context = ocl_getContext();

  Datum values[kde_table_num_rels];
  bool  nulls[kde_table_num_rels];
  bool  repl[kde_table_num_rels];
  memset(values, 0, sizeof(values));
  memset(nulls, false, sizeof(nulls));
  memset(repl, true, sizeof(repl));

  // Collect all tuple values.
    // Insert all static fields.
  values[Anum_kde_table_relid-1] = ObjectIdGetDatum(estimator->relation_id);
  values[Anum_kde_table_columns-1] = Int32GetDatum(estimator->columns);
  values[Anum_kde_table_sample_size-1] = Int32GetDatum(estimator->sample_size);
  values[Anum_kde_table_is_exact-1] = BoolGetDatum(estimator->exact);
    // Insert the scale factors.
  array_datums = palloc(sizeof(Datum) * estimator->nr_of_dimensions);
  for (i = 0; i < estimator->nr_of_dimensions; ++i) {
    array_datums[i] = Float4GetDatum(estimator->scale_factors[i]);
  }
  array = construct_array(array_datums, estimator->nr_of_dimensions,
                          FLOAT4OID, sizeof(float), FLOAT4PASSBYVAL, 'i');
  values[Anum_kde_table_scalefactors-1] = PointerGetDatum(array);
    // Insert the bandwidth.
  if (estimator->exact) {
    // We don't have to store the bandwidth.
    nulls[Anum_kde_table_bandwidth-1] = true;
  } else {
    float* host_bandwidth = palloc(estimator->nr_of_dimensions * sizeof(float));
    err = clEnqueueReadBuffer(
        context->queue, estimator->bandwidth_buffer, CL_TRUE, 0,
        estimator->nr_of_dimensions * sizeof(float), host_bandwidth,
        0, NULL, NULL);
    for (i = 0; i < estimator->nr_of_dimensions; ++i) {
      array_datums[i] = Float4GetDatum(host_bandwidth[i]);
    }
    pfree(host_bandwidth);
    array = construct_array(array_datums, estimator->nr_of_dimensions,
                            FLOAT4OID, sizeof(float), FLOAT4PASSBYVAL, 'i');
    values[Anum_kde_table_bandwidth-1] = PointerGetDatum(array);
  }
  // Transfer the sample to a host buffer.
  bytea* host_sample = palloc(
      sizeof(float) * estimator->nr_of_dimensions * estimator->sample_size + VARHDRSZ);
  err = clEnqueueReadBuffer(
      context->queue, estimator->data_sample, CL_TRUE, 0,
      sizeof(float) * estimator->nr_of_dimensions * estimator->sample_size,
      VARDATA(host_sample), 0, NULL, NULL);
  SET_VARSIZE(host_sample, sizeof(float) * estimator->nr_of_dimensions * estimator->sample_size + VARHDRSZ);
  values[Anum_kde_table_sample-1] = PointerGetDatum(host_sample);

  // Now, try to find whether this estimator already exists.
  Relation kdeRel = heap_open(KDE_TABLE_OID, RowExclusiveLock);
  ScanKeyData key[1];
  ScanKeyInit(&key[0], Anum_kde_table_relid, BTEqualStrategyNumber, F_OIDEQ,
              ObjectIdGetDatum(estimator->relation_id));
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
  pfree(array_datums);
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
  if (estimator->data_sample)
    clReleaseMemObject(estimator->data_sample);
  if (estimator->bandwidth_buffer)
    clReleaseMemObject(estimator->bandwidth_buffer);
  if (estimator->range_kernel)
    clReleaseKernel(estimator->range_kernel);
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
KDEReleaseCallback(int code, void* arg) {
  fprintf(stderr, "Releasing OpenCL Resources.\n");
  ocl_releaseRegistry();
  ocl_releaseContext();
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
	Relation kdeRel = heap_open(KDE_TABLE_OID, RowExclusiveLock);
	HeapScanDesc scan = heap_beginscan(kdeRel, SnapshotNow, 0, NULL);
	HeapTuple tuple;
	while ((tuple = heap_getnext(scan, ForwardScanDirection)) != NULL) {
	  ocl_estimator_t* estimator = ocl_buildEstimatorFromCatalogEntry(
	      kdeRel, tuple);
	  if (estimator == NULL) continue;
	  // Note: Right now, we only support a single estimator per table.
	  directory_insert(registry->estimator_directory, &(estimator->relation_id),
	                   estimator);
	  registry->estimator_bitmap[estimator->relation_id / 8]
	                             |= (0x1 << estimator->relation_id % 8);
	}
	heap_endscan(scan);
	heap_close(kdeRel, RowExclusiveLock);
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
		clEnqueueWriteBuffer(ctxt->queue, ctxt->input_buffer, CL_FALSE, 0, 
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

int ocl_prepareEstimator(ocl_estimator_t* estimator) {
	ocl_context_t* ctxt = ocl_getContext();
	if (ctxt == NULL) return 0;
	// Construct the build-string
	char build_string[40];
	snprintf(build_string, 40, "-DD=%i -DSAMPLE_SIZE=%i",
	         estimator->nr_of_dimensions, estimator->sample_size);
	// Prepare the kernel. If we have a full sample, use the exact kernel, otherwise use KDE.
	if (estimator->exact) {
		estimator->range_kernel = ocl_getKernel("exact_kde", build_string);
   } else {
		estimator->range_kernel = ocl_getKernel("range_kde", build_string);
	}
	// Make sure the kernel was correctly created.
	if (estimator->range_kernel == NULL) {
		return 0;
	}
	cl_int err = 0;
	if (!estimator->exact) {
		err |= clSetKernelArg(estimator->range_kernel, 3, sizeof(cl_mem), &(estimator->bandwidth_buffer));
	}
	// Now prepare the arguments.
	err |= clSetKernelArg(estimator->range_kernel, 0, sizeof(cl_mem), &(estimator->data_sample));
	err |= clSetKernelArg(estimator->range_kernel, 1, sizeof(cl_mem), &(ctxt->result_buffer));
	err |= clSetKernelArg(estimator->range_kernel, 2, sizeof(cl_mem), &(ctxt->input_buffer));
	if (err) {
		clReleaseKernel(estimator->range_kernel);
		if (!estimator->exact) clReleaseMemObject(estimator->bandwidth_buffer);
		return 0;
	}
	// And compute the range normalization factor:
	if (estimator->exact)
		estimator->range_normalization_factor = 1.0f /(float)estimator->sample_size;
	else	
		estimator->range_normalization_factor = 0.75f /(float)estimator->sample_size;
	return 1;
}


unsigned int ocl_maxSampleSize(unsigned int dimensionality) {
	return (kde_samplesize*1024)/(dimensionality*sizeof(float));
}

void ocl_constructEstimator(
    Relation rel, unsigned int rows_in_table, unsigned int dimensionality,
    AttrNumber* attributes, unsigned int sample_size, HeapTuple* sample) {
	unsigned int i, j;
	cl_int err;
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
	estimator->relation_id = rel->rd_node.relNode;
  estimator->nr_of_dimensions = dimensionality;
  estimator->column_order = calloc(1, dimensionality * sizeof(AttrNumber));
	for (i = 0; i<dimensionality; ++i) {
	  estimator->columns |= 0x1 << attributes[i];
	  estimator->column_order[i] = attributes[i];
	}
	estimator->sample_size = sample_size;
	/* 
    * Ok, we set up the estimator. Prepare the sample for shipping it to the device. 
    * While preparing the sample, we compute the variance for each column on the fly.
 	 * The variance is then used to normalize the data to unit variance in each dimension.
    */
	float* host_buffer = (float*)malloc(sizeof(float)*dimensionality*sample_size);
	estimator->scale_factors = (float*)malloc(sizeof(float)*dimensionality);
	for ( i = 0; i < dimensionality; ++i) {
		bool isnull;
		// Values for the online variance computation	
		unsigned int n = 0;
		float mean = 0;
		float M2 = 0;
		// Make sure we use the correct extraction function for both double and float.
		for ( j = 0; j < sample_size; ++j) {
			Oid attribute_type = rel->rd_att->attrs[attributes[i]-1]->atttypid;
			float v = 0.0f;
			if (attribute_type == FLOAT4OID)
				v = DatumGetFloat4(heap_getattr(sample[j], attributes[i], rel->rd_att, &isnull));
			else if (attribute_type == FLOAT8OID)
				v = DatumGetFloat8(heap_getattr(sample[j], attributes[i], rel->rd_att, &isnull));
			// Store in the host buffer. Adjust for access patterns on the device.
			host_buffer[j*dimensionality + i] = v;
			// And update the variance
			n++;
			float delta = v - mean;
			mean += delta / n;
			M2 += delta*(v - mean);		
		}
		estimator->scale_factors[i] = sqrt(M2/(n-1));
	}
	// Re-scale the data to unit variance
	for ( j = 0; j < sample_size; ++j) {
		for ( i = 0; i < dimensionality; ++i) {
			host_buffer[j*dimensionality + i] /= estimator->scale_factors[i];
		}
	}
	// Now compute an initial bandwidth estimate using scott's rule
	float* bandwidth = (float*)malloc(sizeof(float)*dimensionality);
	for ( i = 0; i < dimensionality; ++i) {
		bandwidth[i] = pow(sample_size, -1.0f/(float)(dimensionality+4))*sqrt(5);
	}
	cl_mem bandwidth_buffer = clCreateBuffer(
	    ctxt->context, CL_MEM_READ_WRITE,
	    sizeof(float)*dimensionality, NULL, NULL);
  clEnqueueWriteBuffer(
      ctxt->queue, bandwidth_buffer, CL_TRUE, 0,
      sizeof(float)*dimensionality, bandwidth, 0, NULL, NULL);
	estimator->bandwidth_buffer = bandwidth_buffer;
	// Print some debug info
	fprintf(stderr, "\tBandwidth:");
	for ( i = 0; i< dimensionality ; ++i)
		fprintf(stderr, " %f", bandwidth[i]);
	fprintf(stderr, "\n"); 
	// Push the sample to the device.
	cl_mem sample_buffer = clCreateBuffer(
	    ctxt->context, CL_MEM_READ_WRITE,
	    sizeof(float)*dimensionality*sample_size, NULL, NULL);
	err = clEnqueueWriteBuffer(
	    ctxt->queue, sample_buffer, CL_TRUE, 0,
	    sizeof(float)*dimensionality*sample_size, host_buffer, 0, NULL, NULL);
  free(host_buffer);
	estimator->data_sample = sample_buffer;
	// Now prepare and register the estimator.
	estimator->exact = (rows_in_table == sample_size);
	ocl_prepareEstimator(estimator);
	ocl_updateEstimatorInCatalog(estimator);

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

#endif /* USE_OPENCL */
