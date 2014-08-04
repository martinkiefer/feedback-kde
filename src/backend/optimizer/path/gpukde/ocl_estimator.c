#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
/*

 * ocl_estimator.c
 *
 *  Created on: 25.05.2012
 *      Author: mheimel
 */

#include "ocl_adaptive_bandwidth.h"
#include "ocl_estimator.h"
#include "ocl_model_maintenance.h"
#include "ocl_utilities.h"

#ifdef USE_OPENCL

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#include "miscadmin.h"
#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/sysattr.h"
#include "access/xact.h"
#include "catalog/pg_kdemodels.h"
#include "catalog/pg_type.h"
#include "storage/lock.h"
#include "storage/ipc.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/rel.h"
#include "utils/tqual.h"

extern bool ocl_use_gpu;
extern bool kde_enable;
extern int kde_samplesize;

ocl_kernel_type_t global_kernel_type = GAUSS;
bool scale_to_unit_variance = false;

/*
 * Global registry variable
 */
ocl_estimator_registry_t* registry = NULL;

static cl_kernel prepareKDEKernel(
    ocl_estimator_t* estimator, const char* kernel_name) {
  // First, fetch the correct program.
  cl_kernel kernel = ocl_getKernel(kernel_name,
                                   estimator->nr_of_dimensions);
  // Now configure the kernels.
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &(estimator->sample_buffer));
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &(ocl_getContext()->result_buffer));
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &(ocl_getContext()->input_buffer));
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &(estimator->bandwidth_buffer));
  return kernel;
}

static double rangeKDE(ocl_context_t* ctxt, ocl_estimator_t* estimator) {
  // Select kernel and normalization factor based on the kernel type.
  cl_kernel kde_kernel;
  kde_float_t normalization_factor = 1.0;
  if (global_kernel_type == EPANECHNIKOV) {
    // Epanechnikov.
    if (estimator->estimator == NULL)
      estimator->estimator = prepareKDEKernel(estimator, "epanechnikov_kde");
    kde_kernel = estimator->estimator;
    // (3/4)^d
    normalization_factor = pow(0.75, estimator->nr_of_dimensions);
  } else {
    // Gauss.
    if (estimator->estimator == NULL)
      estimator->estimator = prepareKDEKernel(estimator, "gauss_kde");
    kde_kernel = estimator->estimator;
    // (1/2)^d
    normalization_factor = pow(0.5, estimator->nr_of_dimensions);
  }
  // Now run the actual computation.
  size_t global_size = estimator->rows_in_sample;
  cl_event kde_event;
  if (estimator->online_learning_event) {
	  clEnqueueNDRangeKernel(ctxt->queue, kde_kernel, 1, NULL, &global_size,
	                         NULL, 1, &(estimator->online_learning_event),
                           &kde_event);
	  clReleaseEvent(estimator->online_learning_event);
	  estimator->online_learning_event = NULL;
  } else {
	  clEnqueueNDRangeKernel(ctxt->queue, kde_kernel, 1, NULL, &global_size,
	                         NULL, 0, NULL, &kde_event);
  }
  cl_mem result_buffer = clCreateBuffer(
	    ctxt->context, CL_MEM_READ_WRITE, sizeof(kde_float_t), NULL, NULL);
  cl_event sum_event = sumOfArray(
	    ctxt->result_buffer, estimator->rows_in_sample, result_buffer, 0,
	    kde_event);
  kde_float_t result;
  clEnqueueReadBuffer(ctxt->queue, result_buffer, CL_TRUE, 0,
	                    sizeof(kde_float_t), &result, 1, &sum_event, NULL);
  result *= normalization_factor / estimator->rows_in_sample;
  clReleaseMemObject(result_buffer);
  clReleaseEvent(kde_event);
  clReleaseEvent(sum_event);
  return result;
}

static ocl_estimator_t* ocl_buildEstimatorFromCatalogEntry(
    Relation kde_rel, HeapTuple tuple) {
  unsigned int i,j;
  Datum datum;
  ArrayType* array;
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
  descriptor->column_order = calloc(1, 32*sizeof(unsigned int));
  for ( i=0; columns && i<32; ++i ) {
    if (columns & (0x1)) {
      descriptor->column_order[i] = descriptor->nr_of_dimensions++;
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
      sizeof(double) * descriptor->nr_of_dimensions);
  memcpy(descriptor->scale_factors, (float8*)ARR_DATA_PTR(array),
         descriptor->nr_of_dimensions * sizeof(float8));

  // >> Read the bandwidth.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_bandwidth,
                       RelationGetDescr(kde_rel), &isNull);
  array = DatumGetArrayTypeP(datum);
  descriptor->bandwidth_buffer = clCreateBuffer(
          context->context, CL_MEM_READ_WRITE,
          sizeof(kde_float_t)*descriptor->nr_of_dimensions, NULL, NULL);
  if (sizeof(kde_float_t) == sizeof(float)) {
    // The system catalog stores double, but we expect float.
    kde_float_t* tmp_buffer = palloc(sizeof(kde_float_t) * descriptor->nr_of_dimensions);
    float8* catalog_ptr = (float8*)ARR_DATA_PTR(array);
    for ( i=0; i<descriptor->nr_of_dimensions; ++i ) {
      tmp_buffer[i] = catalog_ptr[i];
    }
    clEnqueueWriteBuffer(context->queue, descriptor->bandwidth_buffer, CL_TRUE,
                         0, sizeof(kde_float_t)*descriptor->nr_of_dimensions,
                         tmp_buffer, 0, NULL, NULL);
    pfree(tmp_buffer);
  } else {
    clEnqueueWriteBuffer(context->queue, descriptor->bandwidth_buffer, CL_FALSE,
                         0, sizeof(kde_float_t)*descriptor->nr_of_dimensions,
                         (char*)ARR_DATA_PTR(array), 0, NULL, NULL);
  }

  // >> Read the sample.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_sample_file,
                       RelationGetDescr(kde_rel), &isNull);
  char* file_name = TextDatumGetCString(datum);
  FILE* file = fopen(file_name, "rb");
  if (file == NULL) {
    fprintf(stderr, "Error opening sample file %s\n", file_name);
    return NULL;
  }
  double* sample_buffer = palloc(
      sizeof(double) * descriptor->nr_of_dimensions * descriptor->rows_in_sample);
  size_t read_elements = fread(
      sample_buffer, sizeof(double) * descriptor->nr_of_dimensions,
      descriptor->rows_in_sample, file);
  if (read_elements != descriptor->rows_in_sample) {
    fprintf(stderr, "Error reading sample from file %s\n", file_name);
    fclose(file);
    pfree(sample_buffer);
    return NULL;
  }
  // Read the sample karma.
  double* karma_buffer = palloc(
      sizeof(double) * descriptor->rows_in_sample);
  read_elements = fread(
      karma_buffer, sizeof(double),
      descriptor->rows_in_sample, file);
  if (read_elements != descriptor->rows_in_sample) {
    fprintf(stderr, "Error reading sample from file %s\n", file_name);
    fclose(file);
    pfree(sample_buffer);
    return NULL;
  }
  // Read the sample contribution.
  double* contribution_buffer = palloc(
      sizeof(double) * descriptor->rows_in_sample);
  read_elements = fread(
      contribution_buffer, sizeof(double),
      descriptor->rows_in_sample, file);
  if (read_elements != descriptor->rows_in_sample) {
    fprintf(stderr, "Error reading sample from file %s\n", file_name);
    fclose(file);
    pfree(sample_buffer);
    return NULL;
  }
  fclose(file);
  
  // Now build and initialize the required OpenCL buffers.
  descriptor->sample_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      descriptor->sample_buffer_size, NULL, NULL);
  descriptor->sample_karma_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      descriptor->rows_in_sample*sizeof(kde_float_t), NULL, NULL);
  descriptor->sample_contribution_buffer= clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      descriptor->rows_in_sample*sizeof(kde_float_t), NULL, NULL);
  if (sizeof(kde_float_t) == sizeof(float)) {
    kde_float_t* sample_transfer_buffer = palloc(
        sizeof(kde_float_t) * descriptor->nr_of_dimensions *
        descriptor->rows_in_sample);
    kde_float_t* karma_transfer_buffer = palloc(
        sizeof(kde_float_t) * descriptor->rows_in_sample);
    kde_float_t* contribution_transfer_buffer = palloc(
        sizeof(kde_float_t) * descriptor->rows_in_sample);
    for( j=0; j < descriptor->rows_in_sample; ++j){
      karma_transfer_buffer[j] = karma_buffer[j];
      contribution_transfer_buffer[j] = contribution_buffer[j];
      for ( i=0; i<descriptor->nr_of_dimensions; ++i ) {
        sample_transfer_buffer[j*descriptor->nr_of_dimensions+i] =
            sample_buffer[j*descriptor->nr_of_dimensions+i];
      }
    }
    clEnqueueWriteBuffer(
        context->queue, descriptor->sample_buffer, CL_TRUE, 0,
        ocl_sizeOfSampleItem(descriptor) * descriptor->rows_in_sample,
        sample_transfer_buffer, 0, NULL, NULL);
    clEnqueueWriteBuffer(
        context->queue, descriptor->sample_karma_buffer, CL_TRUE, 0,
        sizeof(kde_float_t) * descriptor->rows_in_sample,
        karma_transfer_buffer, 0, NULL, NULL);
    clEnqueueWriteBuffer(
        context->queue, descriptor->sample_contribution_buffer, CL_TRUE, 0,
        sizeof(kde_float_t) * descriptor->rows_in_sample,
        contribution_transfer_buffer, 0, NULL, NULL);
    pfree(sample_transfer_buffer);
    pfree(karma_transfer_buffer);
    pfree(contribution_transfer_buffer);
  } else if (sizeof(kde_float_t) == sizeof(double)) {
    clEnqueueWriteBuffer(
        context->queue, descriptor->sample_buffer, CL_TRUE, 0,
        ocl_sizeOfSampleItem(descriptor) * descriptor->rows_in_sample,
        sample_buffer, 0, NULL, NULL);
    clEnqueueWriteBuffer(
        context->queue, descriptor->sample_karma_buffer, CL_TRUE, 0,
        sizeof(kde_float_t) * descriptor->rows_in_sample,
        karma_buffer, 0, NULL, NULL);
    clEnqueueWriteBuffer(
        context->queue, descriptor->sample_contribution_buffer, CL_TRUE, 0,
        sizeof(kde_float_t) * descriptor->rows_in_sample,
        contribution_buffer, 0, NULL, NULL);
  }
  pfree(sample_buffer);
  pfree(karma_buffer);
  pfree(contribution_buffer);
  // Wait for all transfers to finish.
  clFinish(context->queue);
  // We are done.
  return descriptor;
}

static void ocl_updateEstimatorInCatalog(ocl_estimator_t* estimator) {
  unsigned int i,j;
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
  for ( i = 0; i < estimator->nr_of_dimensions; ++i ) {
    array_datums[i] = Float8GetDatum(estimator->scale_factors[i]);
  }
  array = construct_array(array_datums, estimator->nr_of_dimensions,
                          FLOAT8OID, sizeof(double), FLOAT8PASSBYVAL, 'i');
  values[Anum_pg_kdemodels_scale_factors-1] = PointerGetDatum(array);

  // >> Write the bandwidth.
  kde_float_t* host_bandwidth = palloc(
      estimator->nr_of_dimensions * sizeof(kde_float_t));
  clEnqueueReadBuffer(
      context->queue, estimator->bandwidth_buffer, CL_TRUE, 0,
      estimator->nr_of_dimensions * sizeof(kde_float_t), host_bandwidth,
      0, NULL, NULL);
  for (i = 0; i < estimator->nr_of_dimensions; ++i) {
    array_datums[i] = Float8GetDatum(host_bandwidth[i]);
  }
  pfree(host_bandwidth);
  array = construct_array(array_datums, estimator->nr_of_dimensions,
                          FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'i');
  values[Anum_pg_kdemodels_bandwidth-1] = PointerGetDatum(array);

  // >> Write the sample to a file.
  kde_float_t* sample_buffer = palloc(
      ocl_sizeOfSampleItem(estimator) * estimator->rows_in_sample);
  kde_float_t* karma_buffer = palloc(
      sizeof(kde_float_t) * estimator->rows_in_sample);
  kde_float_t* contribution_buffer = palloc(
      sizeof(kde_float_t) * estimator->rows_in_sample);
  clEnqueueReadBuffer(
      context->queue, estimator->sample_buffer, CL_TRUE, 0,
      ocl_sizeOfSampleItem(estimator) * estimator->rows_in_sample,
      sample_buffer, 0, NULL, NULL);
  clEnqueueReadBuffer(
      context->queue, estimator->sample_karma_buffer, CL_TRUE, 0,
      sizeof(kde_float_t) * estimator->rows_in_sample,
      karma_buffer, 0, NULL, NULL);
  clEnqueueReadBuffer(
      context->queue, estimator->sample_contribution_buffer, CL_TRUE, 0,
      sizeof(kde_float_t) * estimator->rows_in_sample,
      contribution_buffer, 0, NULL, NULL);
  // Open the sample file for this table.
  char sample_file_name[1024];
  sprintf(sample_file_name, "%s/pg_kde_samples/rel%i_kde.sample",
          DataDir, estimator->table);
  FILE* sample_file = fopen(sample_file_name, "wb");
  // Now store all buffers to the location.
  if (sizeof(kde_float_t) == sizeof(double)) {
    fwrite(sample_buffer, sizeof(kde_float_t)*estimator->nr_of_dimensions,
           estimator->rows_in_sample, sample_file);
    fwrite(karma_buffer, sizeof(kde_float_t),
           estimator->rows_in_sample, sample_file);
    fwrite(contribution_buffer, sizeof(kde_float_t),
           estimator->rows_in_sample, sample_file);
  } else {
    double* sample_transfer_buffer = palloc(
        sizeof(double) * estimator->nr_of_dimensions * estimator->rows_in_sample);
    double* karma_transfer_buffer = palloc(
        sizeof(double) * estimator->rows_in_sample);
    double* contribution_transfer_buffer = palloc(
        sizeof(double) * estimator->rows_in_sample);
    for( j=0; j < estimator->rows_in_sample; ++j){
      karma_transfer_buffer[j] = karma_buffer[j];
      contribution_transfer_buffer[j] = contribution_buffer[j];
      for ( i=0; i<estimator->nr_of_dimensions; ++i ) {
        sample_transfer_buffer[j*estimator->nr_of_dimensions+i] =
            sample_buffer[j*estimator->nr_of_dimensions+i];
      }
    }  
    fwrite(sample_transfer_buffer, sizeof(double)*estimator->nr_of_dimensions,
           estimator->rows_in_sample, sample_file);
    fwrite(karma_transfer_buffer, sizeof(double),
           estimator->rows_in_sample, sample_file);
    fwrite(contribution_transfer_buffer, sizeof(double),
           estimator->rows_in_sample, sample_file);
    pfree(sample_transfer_buffer);
    pfree(karma_transfer_buffer);
    pfree(contribution_transfer_buffer);
  }
  fclose(sample_file);
  pfree(sample_buffer);
  pfree(karma_buffer);
  pfree(contribution_buffer);
  values[Anum_pg_kdemodels_sample_file-1] = CStringGetTextDatum(sample_file_name);

  // Ok, we constructed the tuple. Now try to find whether the estimator is
  // already present in the catalog.
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
    
    HeapTuple newtuple = heap_modify_tuple(tuple, RelationGetDescr(kdeRel),
                                           values, nulls, repl);
    
    simple_heap_update(kdeRel, &tuple->t_self, newtuple);
    heap_endscan(scan);
    
  }
  heap_close(kdeRel, RowExclusiveLock);

  // Clean up.
  pfree(array_datums);
}

/*
 *  Static helper function to release the resources held by a single estimator.
 *
 *  If materialize is set, the function will write the changes to stable
 *  storage.
 */
static void ocl_freeEstimator(ocl_estimator_t* estimator, bool materialize) {
  if (estimator == NULL) return;
  // Remove the estimator from the registry.
  if (registry) {
    registry->estimator_bitmap[estimator->table / 8] ^= (0x1 << (estimator->table % 8));
  }
  // Write all changes to stable storage.
  if (materialize) ocl_updateEstimatorInCatalog(estimator);
  // Release the in-memory structure.
  if (estimator->scale_factors)
    free(estimator->scale_factors);
  if (estimator->sample_buffer)
    clReleaseMemObject(estimator->sample_buffer);
  if(estimator->sample_karma_buffer)
    clReleaseMemObject(estimator->sample_karma_buffer);
  if(estimator->sample_contribution_buffer)
    clReleaseMemObject(estimator->sample_contribution_buffer);
  if (estimator->bandwidth_buffer)
    clReleaseMemObject(estimator->bandwidth_buffer);
    // Release all buffers for the online optimization.
  if (estimator->gradient_accumulator)
    clReleaseMemObject(estimator->gradient_accumulator);
  if (estimator->squared_gradient_accumulator)
    clReleaseMemObject(estimator->squared_gradient_accumulator);
  if (estimator->hessian_accumulator)
    clReleaseMemObject(estimator->hessian_accumulator);
  if (estimator->squared_hessian_accumulator)
    clReleaseMemObject(estimator->squared_hessian_accumulator);
  if (estimator->running_gradient_average)
    clReleaseMemObject(estimator->running_gradient_average);
  if (estimator->running_squared_gradient_average)
    clReleaseMemObject(estimator->running_squared_gradient_average);
  if (estimator->running_hessian_average)
    clReleaseMemObject(estimator->running_hessian_average);
  if (estimator->running_squared_hessian_average)
    clReleaseMemObject(estimator->running_squared_hessian_average);
  if (estimator->current_time_constant)
    clReleaseMemObject(estimator->current_time_constant);
  if (estimator->temp_gradient_buffer)
    clReleaseMemObject(estimator->temp_gradient_buffer);
  if (estimator->temp_shifted_gradient_buffer)
    clReleaseMemObject(estimator->temp_shifted_gradient_buffer);
  if (estimator->temp_shifted_result_buffer)
    clReleaseMemObject(estimator->temp_shifted_result_buffer);

  if (estimator->column_order)
    free(estimator->column_order);
  free(estimator);
}

static void ocl_releaseRegistry() {
  if (!registry) return;
  unsigned int i;
  // Update all registered estimators within the system catalogue.
  for (i=0; i<registry->estimator_directory->entries; ++i) {
    ocl_estimator_t* estimator = (ocl_estimator_t*)directory_valueAt(
        registry->estimator_directory, i);
    ocl_freeEstimator(estimator,true);
  }
  // Now release the registry.
  directory_release(registry->estimator_directory, false);
  free(registry);
  registry = NULL;
}

static void
ocl_cleanUpRegistry(int code, Datum arg) {
  if (!registry) return;
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
	registry->estimator_bitmap = calloc(1, 512*1024); // Enough bits for ~4M tables.
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
    // Register the estimator.
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

// Helper function to print a request to stderr.
static void ocl_dumpRequest(const ocl_estimator_request_t* request) {
	unsigned int i;
	if (!request) return;
	fprintf(stderr, "Received estimation request for table: %i:\n",
	        request->table_identifier);
	for (i=0; i<request->range_count; ++i)
		fprintf(stderr, "\tColumn %i in: [%f , %f]\n", request->ranges[i].colno,
		        request->ranges[i].lower_bound, request->ranges[i].upper_bound);
}

int ocl_updateRequest(ocl_estimator_request_t* request,
		AttrNumber colno, double* lower_bound, bool lower_included,
		double* upper_bound, bool upper_included) {
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
		request->ranges->lower_bound = -1.0 * INFINITY;
		request->ranges->upper_bound = INFINITY;
		column_range = request->ranges;
	} else {
		/* Check whether we already have a range for this column */
		column_range = bsearch(
				&colno, request->ranges, request->range_count,
				sizeof(ocl_colrange_t), &compareRange);
		if (column_range == NULL) {
			/* We have to add the column. Add storage for a new value */
			request->range_count++;
			request->ranges = (ocl_colrange_t*)realloc(
			    request->ranges, sizeof(ocl_colrange_t)*(request->range_count));
			/* Initialize the new column range */
			column_range = &(request->ranges[request->range_count-1]);
			column_range->colno = colno;
			column_range->lower_bound = -1.0 * INFINITY;
			column_range->upper_bound = INFINITY;
			/* Now we have to re-sort the array */
			qsort(request->ranges, request->range_count,
			      sizeof(ocl_colrange_t), &compareRange);
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
	column_range->lower_included = lower_included;
	column_range->upper_included = upper_included;
	return 1;
}

int ocl_estimateSelectivity(const ocl_estimator_request_t* request,
		Selectivity* selectivity) {
	struct timeval start; gettimeofday(&start, NULL); 
	unsigned int i;
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
	kde_float_t* row_ranges = (kde_float_t*)malloc(
	    2*sizeof(kde_float_t)*estimator->nr_of_dimensions);
	for (i = 0; i < estimator->nr_of_dimensions; ++i) {
		row_ranges[2*i] = -1.0 * INFINITY;
		row_ranges[2*i+1] = INFINITY;
	}
	for (i = 0; i < request->range_count; ++i) {
		unsigned int range_pos = estimator->column_order[request->ranges[i].colno];
		row_ranges[2*range_pos] =
		request->ranges[i].lower_bound * estimator->scale_factors[range_pos];
		row_ranges[2*range_pos + 1] =
		request->ranges[i].upper_bound * estimator->scale_factors[range_pos];
		if (request->ranges[i].lower_included)
			row_ranges[2*range_pos] -= 0.001;
	if (request->ranges[i].upper_included)
		row_ranges[2*range_pos + 1] += 0.001;
	}
	// Prepare the request.
	clEnqueueWriteBuffer(ctxt->queue, ctxt->input_buffer, CL_TRUE, 0,
               2*estimator->nr_of_dimensions*sizeof(kde_float_t), row_ranges,
               0, NULL, NULL);
	*selectivity = rangeKDE(ctxt, estimator);
	estimator->last_selectivity = *selectivity;
	estimator->open_estimation = true;
	// Print timing:
	struct timeval now; gettimeofday(&now, NULL);
	long seconds = now.tv_sec - start.tv_sec;
	long useconds = now.tv_usec - start.tv_usec;
	long mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	if (ocl_isDebug()) {
		ocl_dumpRequest(request);
		fprintf(stderr, "Estimated selectivity: %f, took: %ld ms.\n", *selectivity, mtime);
	}
	free(row_ranges);
	// Finally, prepare the online update of the bandwidth.
	ocl_prepareOnlineLearningStep(estimator);
	return 1;
}

unsigned int ocl_maxSampleSize(unsigned int dimensionality) {
	return kde_samplesize;
}

void ocl_constructEstimator(
    Relation rel, unsigned int rows_in_table, unsigned int dimensionality,
    AttrNumber* attributes, unsigned int sample_size, HeapTuple* sample) {
	unsigned int i, j;
	if (dimensionality > 10) {
	  fprintf(stderr, "We only support models for up to 10 dimensions!\n");
	  return;
	}
	// Make sure we have a context
	ocl_context_t* ctxt = ocl_getContext();
	if (ctxt == NULL)	return;
	// Make sure the registry exists.
	if (!registry) ocl_initializeRegistry();
	// Some Debug output
	fprintf(stderr, "Constructing an estimator for table %i.\n",
	        rel->rd_node.relNode);
	fprintf(stderr, "\tColumns:");
	for (i=0; i<dimensionality; ++i) fprintf(stderr, " %i", attributes[i]);
	fprintf(stderr, "\n");
	fprintf(stderr, "\tUsing a backing sample of %i out of %i tuples.\n",
	        sample_size, rows_in_table);
	// Insert a new estimator.
	Assert(rel->rd_node.relNode / 8 >= 16384); //If the oids are to large, bad things will hapen.
	ocl_estimator_t* estimator = calloc(1, sizeof(ocl_estimator_t));
	ocl_estimator_t* old_estimator = directory_insert(
	    registry->estimator_directory, &(rel->rd_node.relNode), estimator);
	// If there was an existing estimator, release it.
	if (old_estimator) {
	  ocl_freeEstimator(old_estimator, false);
	}
	// Update the bitmap.
	registry->estimator_bitmap[rel->rd_node.relNode / 8]
                             |= (0x1 << rel->rd_node.relNode % 8);
	// Update the descriptor info.
	estimator->table = rel->rd_node.relNode;
	estimator->nr_of_dimensions = dimensionality;
	estimator->column_order = calloc(1, 32 * sizeof(unsigned int));
	for (i = 0; i<dimensionality; ++i) {
	  estimator->columns |= 0x1 << attributes[i];
	  estimator->column_order[attributes[i]] = i;
	}
	estimator->rows_in_sample = sample_size;
	estimator->rows_in_table = rows_in_table;
	/* 
    * Ok, we set up the estimator. Prepare the sample for shipping it to the device. 
    * While preparing the sample, we compute the variance for each column on the fly.
 	 * The variance is then used to normalize the data to unit variance in each dimension.
    */
	kde_float_t* host_buffer = (kde_float_t*)malloc(
	    ocl_sizeOfSampleItem(estimator) * sample_size);
	double* mean = (double*)calloc(1, sizeof(double) * dimensionality);
	double* M2 = (double*)calloc(1, sizeof(double) * dimensionality);
	for ( i = 0; i < sample_size; ++i ) {
	  // Extract the item.
	  ocl_extractSampleTuple(estimator, rel, sample[i],
	                         &(host_buffer[i*estimator->nr_of_dimensions]));
	  // And update the attributes.
	  for ( j = 0; j < estimator->nr_of_dimensions; ++j ) {
	    double delta = host_buffer[i*estimator->nr_of_dimensions + j] - mean[j];
	    mean[j] = mean[j] + delta / (i + 1);
	    M2[j] += delta * (host_buffer[i*estimator->nr_of_dimensions + j] - mean[j]);
	  }
	}
	// Compute the scale factors (this is just the standard deviation per dimension).
	free(mean);
	estimator->scale_factors = M2;
	for ( i = 0; i < estimator->nr_of_dimensions; ++i ) {
	  estimator->scale_factors[i] /= (sample_size - 1);
	  estimator->scale_factors[i] = sqrt(estimator->scale_factors[i]);
	}
	// Compute an initial bandwidth estimate using Scott's rule.
	kde_float_t* bandwidth = (kde_float_t*)malloc(
	    sizeof(kde_float_t)*dimensionality);
	for ( i = 0; i < dimensionality; ++i ) {
	  if (!scale_to_unit_variance) {
	    // If data scaling is deactivated, the scale factor is simply one - and
	    // the bandwidth is computed from the stdev.
	    bandwidth[i] = estimator->scale_factors[i] *
	        pow(sample_size, -1.0 /((double)(dimensionality + 4)));
	    if (global_kernel_type == EPANECHNIKOV) bandwidth[i] *= sqrt(5);
	    estimator->scale_factors[i] = 1;
	  } else {
	    // If data scaling is activated, we scale the data to unit variance.
	    bandwidth[i] = 100*pow(sample_size, -1.0 /((double)(dimensionality + 4)));
      if (global_kernel_type == EPANECHNIKOV) bandwidth[i] *= sqrt(5);
      estimator->scale_factors[i] = 1.0 / estimator->scale_factors[i];
	  }
	}
	estimator->bandwidth_buffer = clCreateBuffer(
	    ctxt->context, CL_MEM_READ_WRITE, sizeof(kde_float_t) * dimensionality,
	    NULL, NULL);
  clEnqueueWriteBuffer(
      ctxt->queue, estimator->bandwidth_buffer, CL_TRUE, 0,
      sizeof(kde_float_t) * dimensionality, bandwidth, 0, NULL, NULL);
  // Print some debug info.
  fprintf(stderr, "\tInitial bandwidth guess:");
  for ( i = 0; i < dimensionality ; ++i) {
    fprintf(stderr, " %f", bandwidth[i]);
  }
  fprintf(stderr, "\n");
  free(bandwidth);
  // Allocate a buffer of ones to initialize karma and contribution.
  kde_float_t* one_buffer = (kde_float_t*) malloc(
      sizeof(kde_float_t) * sample_size);
  // Re-scale the data to unit variance.
  for ( j = 0; j < sample_size; ++j ) {
    one_buffer[j] = 1.0f;
    for ( i = 0; i < dimensionality; ++i ) {
      host_buffer[j*dimensionality + i] *= estimator->scale_factors[i];
    }
  }
  // Push the sample to the device.
  estimator->sample_buffer_size =
      kde_samplesize * estimator->nr_of_dimensions * sizeof(kde_float_t);
  estimator->sample_buffer = clCreateBuffer(
      ctxt->context, CL_MEM_READ_WRITE, estimator->sample_buffer_size,
      NULL, NULL);
  estimator->sample_karma_buffer = clCreateBuffer(
      ctxt->context, CL_MEM_READ_WRITE, estimator->sample_buffer_size,
      NULL, NULL);
  estimator->sample_contribution_buffer = clCreateBuffer(
      ctxt->context, CL_MEM_READ_WRITE, estimator->sample_buffer_size,
      NULL, NULL);
	clEnqueueWriteBuffer(
	    ctxt->queue, estimator->sample_buffer, CL_TRUE, 0,
	    sample_size * ocl_sizeOfSampleItem(estimator), host_buffer,
	    0, NULL, NULL);
	clEnqueueWriteBuffer(
	    ctxt->queue, estimator->sample_karma_buffer, CL_TRUE, 0,
	    sample_size * sizeof(kde_float_t), one_buffer,
	    0, NULL, NULL);
  clEnqueueWriteBuffer(
      ctxt->queue, estimator->sample_contribution_buffer, CL_TRUE, 0,
      sample_size * sizeof(kde_float_t), one_buffer,
      0, NULL, NULL);
  free(host_buffer);
  free(one_buffer);
    // Wait for the initialization to finish.
  clFinish(ocl_getContext()->queue);

  // Finally, hand the estimator over for model optimization.
  estimator->learning_boost_rate = 10;
  ocl_runModelOptimization(estimator);
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

void assign_kde_enable(bool newval, void *extra) {
  if (newval != kde_enable) {
    ocl_releaseRegistry();
    ocl_releaseContext();
  }
}

bool ocl_useKDE(void) {
  return kde_enable;
}

ocl_estimator_t* ocl_getEstimator(Oid relation) {
  if (!ocl_useKDE()){
	return NULL;
  }
  ocl_estimator_registry_t* registry = ocl_getRegistry();
  if (registry == NULL){
	return NULL;
  }
  // Check the bitmap whether we have an estimator for this relation.
  if (!(registry->estimator_bitmap[relation / 8]
          & (0x1 << (relation % 8)))){
	return NULL;
  }
  return DIRECTORY_FETCH(registry->estimator_directory, &relation,
                         ocl_estimator_t);
}

size_t ocl_sizeOfSampleItem(ocl_estimator_t* estimator) {
  return estimator->nr_of_dimensions * sizeof(kde_float_t);
}

unsigned int ocl_maxRowsInSample(ocl_estimator_t* estimator) {
  return estimator->sample_buffer_size / ocl_sizeOfSampleItem(estimator);
}


void ocl_pushEntryToSampleBufer(ocl_estimator_t* estimator, int position,
                                kde_float_t* data_item) {
  ocl_context_t* context = ocl_getContext();
  kde_float_t one = 1.0;
  size_t transfer_size = ocl_sizeOfSampleItem(estimator);
  size_t offset = position * transfer_size;
  clEnqueueWriteBuffer(
      context->queue, estimator->sample_buffer, CL_FALSE,
      offset, transfer_size, data_item, 0, NULL, NULL);
  // Initialize the metrics (both to one, so newly sampled items are not immediately replaced).
  clEnqueueWriteBuffer(
      context->queue, estimator->sample_karma_buffer, CL_FALSE,
      position*sizeof(kde_float_t), sizeof(kde_float_t), &one, 0, NULL, NULL);
  clEnqueueWriteBuffer(
      context->queue, estimator->sample_contribution_buffer, CL_FALSE,
      position*sizeof(kde_float_t), sizeof(kde_float_t), &one, 0, NULL, NULL);
  clFinish(context->queue);
}

void ocl_extractSampleTuple(ocl_estimator_t* estimator, Relation rel,
                            HeapTuple tuple, kde_float_t* target) {
  unsigned int i;
  for ( i=0; i<rel->rd_att->natts; ++i ) {
    // Check if this column is contained in the estimator.
    int16 colno = rel->rd_att->attrs[i]->attnum;
    if (!(estimator->columns & (0x1 << colno))) continue;
    // Cool, it is. Check where to write the column content.
    unsigned int wpos = estimator->column_order[colno];
    Oid attribute_type = rel->rd_att->attrs[i]->atttypid;
    bool isNull;
    if (attribute_type == FLOAT4OID)
      target[wpos] = DatumGetFloat4(heap_getattr(tuple, colno, rel->rd_att,
                                                 &isNull));
    else if (attribute_type == FLOAT8OID)
      target[wpos] = DatumGetFloat8(heap_getattr(tuple, colno, rel->rd_att,
                                                 &isNull));
  }
}

#endif /* USE_OPENCL */
