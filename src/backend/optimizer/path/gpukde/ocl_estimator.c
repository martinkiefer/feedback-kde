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
#include "ocl_sample_maintenance.h"
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

// Estimator registration.
ocl_estimator_registry_t* registry = NULL;

// Helper functions to allocate / release an estimator.
static ocl_estimator_t* allocateEstimator(
    Oid relation, int32 column_map, unsigned int sample_size) {
  unsigned int i;
  cl_int err = CL_SUCCESS;
  ocl_estimator_t* result = calloc(1, sizeof(ocl_estimator_t));
  result->table = relation;
  // First, extract the total number of dimensions and the column order from
  // the provided column map.
  result->columns = column_map;
  result->column_order = calloc(1, 32 * sizeof(int));
  for ( i=0; column_map && i<32; ++i ) {
    if (column_map & (0x1)) {
      result->column_order[i] = result->nr_of_dimensions++;
    }
    column_map >>= 1;
  }
  // Now allocate the required buffers.
  ocl_context_t* context = ocl_getContext();
  result->rows_in_sample = sample_size;
  // Allocate the sample buffer.
  result->sample_buffer_size = ocl_sizeOfSampleItem(result) * sample_size;
  result->sample_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      result->sample_buffer_size, NULL, &err);
  Assert(err == CL_SUCCESS);
  // Allocate the buffer to store local results per sample point.
  result->local_results_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * sample_size, NULL, &err);
  Assert(err == CL_SUCCESS);
  // Allocate the buffer to store the final result.
  result->result_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t), NULL, &err);
  Assert(err == CL_SUCCESS);
  // Allocate the input buffer.
  result->input_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      2 * sizeof(kde_float_t) * result->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  // Allocate the bandwidth buffer.
  result->bandwidth_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * result->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  // Prepare the estimation kernel.
  if (global_kernel_type == EPANECHNIKOV) {
    result->kde_kernel = ocl_getKernel(
        "epanechnikov_kde", result->nr_of_dimensions);
  } else {
    result->kde_kernel = ocl_getKernel(
        "gauss_kde", result->nr_of_dimensions);
  }
  err |= clSetKernelArg(
      result->kde_kernel, 0, sizeof(cl_mem), &(result->sample_buffer));
  err |= clSetKernelArg(
      result->kde_kernel, 1, sizeof(cl_mem), &(result->local_results_buffer));
  err |= clSetKernelArg(
      result->kde_kernel, 2, sizeof(cl_mem), &(result->input_buffer));
  err |= clSetKernelArg(
      result->kde_kernel, 3, sizeof(cl_mem), &(result->bandwidth_buffer));
  Assert(err == CL_SUCCESS);
  // Prepare the sum descriptor.
  result->sum_descriptor = prepareSumDescriptor(
      result->local_results_buffer, result->rows_in_sample,
      result->result_buffer, 0);
  // Delegate to allocate the required buffers for the optimization:
  ocl_allocateSampleMaintenanceBuffers(result);
  ocl_allocateBandwidthOptimizatztionBuffers(result);
  return result;
}

static void freeEstimator(ocl_estimator_t* estimator) {
  // Release all buffers.
  cl_int err = CL_SUCCESS;
  if (estimator->sample_buffer) clReleaseMemObject(estimator->sample_buffer);
  if (estimator->local_results_buffer) {
    err = clReleaseMemObject(estimator->local_results_buffer);
    Assert(err == CL_SUCCESS);
  }
  if (estimator->result_buffer) err |= clReleaseMemObject(estimator->result_buffer);
  if (estimator->input_buffer) err |= clReleaseMemObject(estimator->input_buffer);
  if (estimator->bandwidth_buffer) {
    err |= clReleaseMemObject(estimator->bandwidth_buffer);
  }
  Assert(err == CL_SUCCESS);
  
  // Release the kernel.
  if (estimator->kde_kernel) err = clReleaseKernel(estimator->kde_kernel);
  Assert(err == CL_SUCCESS);
  
  // Release the column map.
  if (estimator->column_order) free(estimator->column_order);
  releaseAggregationDescriptor(estimator->sum_descriptor);
  // Release the required buffers for the optimization.
  ocl_releaseSampleMaintenanceBuffers(estimator);
  ocl_releaseBandwidthOptimizatztionBuffers(estimator);
  // Finally, release the estimator itself.
  free(estimator);
}

static ocl_estimator_t* ocl_buildEstimatorFromCatalogEntry(
    Relation kde_rel, HeapTuple tuple) {
  unsigned int i,j;
  cl_int err = CL_SUCCESS;
  Datum datum;
  ArrayType* array;
  bool isNull;
  ocl_context_t* context = ocl_getContext();

  // >> Read table, column map, and sample size.
  datum = heap_getattr(
      tuple, Anum_pg_kdemodels_table, RelationGetDescr(kde_rel), &isNull);
  Oid table = DatumGetObjectId(datum);
  datum = heap_getattr(
      tuple, Anum_pg_kdemodels_columns, RelationGetDescr(kde_rel), &isNull);
  int32 column_map = DatumGetInt32(datum);
  datum = heap_getattr(tuple, Anum_pg_kdemodels_rowcount_sample,
                         RelationGetDescr(kde_rel), &isNull);
  unsigned int sample_size = DatumGetInt32(datum);

  // >> Allocate the descriptor.
  ocl_estimator_t* estimator = allocateEstimator(
      table, column_map, sample_size);

  // >> Read the bandwidth and push it to the device.
  datum = heap_getattr(tuple, Anum_pg_kdemodels_bandwidth,
                       RelationGetDescr(kde_rel), &isNull);
  array = DatumGetArrayTypeP(datum);
  if (sizeof(kde_float_t) == sizeof(float)) {
    // The system catalog stores double, but we expect float.
    kde_float_t* tmp_buffer = palloc(
        sizeof(kde_float_t) * estimator->nr_of_dimensions);
    float8* catalog_ptr = (float8*)ARR_DATA_PTR(array);
    for ( i=0; i<estimator->nr_of_dimensions; ++i ) {
      tmp_buffer[i] = catalog_ptr[i];
    }
    err = clEnqueueWriteBuffer(
        context->queue, estimator->bandwidth_buffer, CL_TRUE,
        0, sizeof(kde_float_t)*estimator->nr_of_dimensions,
        tmp_buffer, 0, NULL, NULL);
    Assert(err == CL_SUCCESS);
    pfree(tmp_buffer);
  } else {
    err = clEnqueueWriteBuffer(
        context->queue, estimator->bandwidth_buffer, CL_FALSE,
        0, sizeof(kde_float_t)*estimator->nr_of_dimensions,
        (char*)ARR_DATA_PTR(array), 0, NULL, NULL);
    Assert(err == CL_SUCCESS);
  }

  // >> Read the sample.
  datum = heap_getattr(
      tuple, Anum_pg_kdemodels_sample_file, RelationGetDescr(kde_rel), &isNull);
  char* file_name = TextDatumGetCString(datum);
  FILE* file = fopen(file_name, "rb");
  if (file == NULL) {
    fprintf(stderr, "Error opening sample file %s\n", file_name);
    return NULL;
  }
  double* sample_buffer = palloc(
      sizeof(double) * estimator->nr_of_dimensions * estimator->rows_in_sample);
  size_t read_elements = fread(
      sample_buffer, sizeof(double) * estimator->nr_of_dimensions,
      estimator->rows_in_sample, file);
  if (read_elements != estimator->rows_in_sample) {
    fprintf(stderr, "Error reading sample from file %s\n", file_name);
    fclose(file);
    pfree(sample_buffer);
    return NULL;
  }
  // Read the sample karma.
  double* karma_buffer = palloc(
      sizeof(double) * estimator->rows_in_sample);
  read_elements = fread(
      karma_buffer, sizeof(double),
      estimator->rows_in_sample, file);
  if (read_elements != estimator->rows_in_sample) {
    fprintf(stderr, "Error reading sample from file %s\n", file_name);
    fclose(file);
    pfree(sample_buffer);
    return NULL;
  }
  // Read the sample contribution.
  double* contribution_buffer = palloc(
      sizeof(double) * estimator->rows_in_sample);
  read_elements = fread(
      contribution_buffer, sizeof(double),
      estimator->rows_in_sample, file);
  if (read_elements != estimator->rows_in_sample) {
    fprintf(stderr, "Error reading sample from file %s\n", file_name);
    fclose(file);
    pfree(sample_buffer);
    return NULL;
  }
  fclose(file);
  
  // Now push sample and sample metrics to the device.
  if (sizeof(kde_float_t) == sizeof(float)) {
    kde_float_t* sample_transfer_buffer = palloc(
        sizeof(kde_float_t) * estimator->nr_of_dimensions *
        estimator->rows_in_sample);
    kde_float_t* karma_transfer_buffer = palloc(
        sizeof(kde_float_t) * estimator->rows_in_sample);
    kde_float_t* contribution_transfer_buffer = palloc(
        sizeof(kde_float_t) * estimator->rows_in_sample);
    for( j=0; j < estimator->rows_in_sample; ++j){
      karma_transfer_buffer[j] = karma_buffer[j];
      contribution_transfer_buffer[j] = contribution_buffer[j];
      for ( i=0; i<estimator->nr_of_dimensions; ++i ) {
        sample_transfer_buffer[j*estimator->nr_of_dimensions+i] =
            sample_buffer[j*estimator->nr_of_dimensions+i];
      }
    }
    err |= clEnqueueWriteBuffer(
        context->queue, estimator->sample_buffer, CL_TRUE, 0,
        ocl_sizeOfSampleItem(estimator) * estimator->rows_in_sample,
        sample_transfer_buffer, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(
        context->queue, estimator->sample_optimization->sample_karma_buffer,
        CL_TRUE, 0, sizeof(kde_float_t) * estimator->rows_in_sample,
        karma_transfer_buffer, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(
        context->queue, estimator->sample_optimization->sample_contribution_buffer,
        CL_TRUE, 0, sizeof(kde_float_t) * estimator->rows_in_sample,
        contribution_transfer_buffer, 0, NULL, NULL);
    Assert(err == CL_SUCCESS);
    
    pfree(sample_transfer_buffer);
    pfree(karma_transfer_buffer);
    pfree(contribution_transfer_buffer);
  } else if (sizeof(kde_float_t) == sizeof(double)) {
    err |= clEnqueueWriteBuffer(
        context->queue, estimator->sample_buffer, CL_TRUE, 0,
        ocl_sizeOfSampleItem(estimator) * estimator->rows_in_sample,
        sample_buffer, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(
        context->queue, estimator->sample_optimization->sample_karma_buffer,
        CL_TRUE, 0, sizeof(kde_float_t) * estimator->rows_in_sample,
        karma_buffer, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(
        context->queue, estimator->sample_optimization->sample_contribution_buffer,
        CL_TRUE, 0, sizeof(kde_float_t) * estimator->rows_in_sample,
        contribution_buffer, 0, NULL, NULL);
    Assert(err == CL_SUCCESS);
  }
  pfree(sample_buffer);
  pfree(karma_buffer);
  pfree(contribution_buffer);
  // Wait for all transfers to finish.
  clFinish(context->queue);
  // We are done.
  return estimator;
}

static void ocl_updateEstimatorInCatalog(ocl_estimator_t* estimator) {
  unsigned int i,j;
  cl_int err = CL_SUCCESS;
  HeapTuple tuple;
  Datum* array_datums = palloc(sizeof(Datum) * estimator->nr_of_dimensions);
  ArrayType* array;
  ocl_context_t* context = ocl_getContext();

  Datum values[Natts_pg_kdemodels];
  bool  nulls[Natts_pg_kdemodels];
  bool  repl[Natts_pg_kdemodels];
  memset(values, 0, sizeof(values));
  memset(nulls, false, sizeof(nulls));
  memset(repl, true, sizeof(repl));
  // We set the scale factors to NULL for now.
  nulls[Anum_pg_kdemodels_scale_factors-1] = true;

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
  err |= clEnqueueReadBuffer(
      context->queue, estimator->sample_buffer, CL_TRUE, 0,
      ocl_sizeOfSampleItem(estimator) * estimator->rows_in_sample,
      sample_buffer, 0, NULL, NULL);
  err |= clEnqueueReadBuffer(
      context->queue, estimator->sample_optimization->sample_karma_buffer,
      CL_TRUE, 0, sizeof(kde_float_t) * estimator->rows_in_sample,
      karma_buffer, 0, NULL, NULL);
  err |= clEnqueueReadBuffer(
      context->queue, estimator->sample_optimization->sample_contribution_buffer,
      CL_TRUE, 0, sizeof(kde_float_t) * estimator->rows_in_sample,
      contribution_buffer, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);
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
  values[Anum_pg_kdemodels_sample_file-1] = CStringGetTextDatum(
      sample_file_name);

  // Ok, we constructed the tuple. Now try to find whether the estimator is
  // already present in the catalog.
  Relation kdeRel = heap_open(KdeModelRelationID, RowExclusiveLock);
  ScanKeyData key[1];
  ScanKeyInit(
      &key[0], Anum_pg_kdemodels_table, BTEqualStrategyNumber, F_OIDEQ,
      ObjectIdGetDatum(estimator->table));
  HeapScanDesc scan = heap_beginscan(kdeRel, SnapshotNow, 1, key);
  tuple = heap_getnext(scan, ForwardScanDirection);
  if (!HeapTupleIsValid(tuple)) {
    // This is a new estimator. Insert it into the table.
    heap_endscan(scan);
    tuple = heap_form_tuple(
        RelationGetDescr(kdeRel), values, nulls);
    simple_heap_insert(kdeRel, tuple);
  } else {
    // This is an existing estimator. Update the corresponding tuple.
    HeapTuple newtuple = heap_modify_tuple(
        tuple, RelationGetDescr(kdeRel), values, nulls, repl);
    simple_heap_update(kdeRel, &tuple->t_self, newtuple);
    heap_endscan(scan);
    
  }
  heap_close(kdeRel, RowExclusiveLock);

  // Clean up.
  pfree(array_datums);
}

// Helper function to compute an actual estimate by the estimator.
static double rangeKDE(
    ocl_context_t* ctxt, ocl_estimator_t* estimator, kde_float_t* query) {
  CREATE_TIMER();
  // Transfer the query bounds to the device.
  cl_event input_transfer_event;
  cl_int err = CL_SUCCESS;
  err = clEnqueueWriteBuffer(
      ctxt->queue, estimator->input_buffer, CL_FALSE,
      0, 2 * sizeof(kde_float_t) * estimator->nr_of_dimensions, query,
      0, NULL, &input_transfer_event);
  Assert(err == CL_SUCCESS);
  // Select kernel and normalization factor based on the kernel type.
  kde_float_t normalization_factor = 1.0;
  if (global_kernel_type == EPANECHNIKOV) {
    // (3/4)^d
    normalization_factor = pow(0.75, estimator->nr_of_dimensions);
  } else {  // Gauss
    // (1/2)^d
    normalization_factor = pow(0.5, estimator->nr_of_dimensions);
  }
  // Compute the local contributions.
  size_t global_size = estimator->rows_in_sample;
  cl_event kde_event;
  if (estimator->bandwidth_optimization->optimization_event) {
    cl_event wait_events[2];
    wait_events[0] = estimator->bandwidth_optimization->optimization_event;
    wait_events[1] = input_transfer_event;
    err = clEnqueueNDRangeKernel(
        ctxt->queue, estimator->kde_kernel, 1, NULL, &global_size,
        NULL, 2, wait_events, &kde_event);
    Assert(err == CL_SUCCESS);
    err = clReleaseEvent(estimator->bandwidth_optimization->optimization_event);
    Assert(err == CL_SUCCESS);
    estimator->bandwidth_optimization->optimization_event = NULL;
  } else {
    err = clEnqueueNDRangeKernel(
        ctxt->queue, estimator->kde_kernel, 1, NULL, &global_size,
        NULL, 1, &input_transfer_event, &kde_event);
    Assert(err == CL_SUCCESS);
  }
  err = clReleaseEvent(input_transfer_event);
  Assert(err == CL_SUCCESS);
  // Compute the final estimation by summing up the local contributions.
  cl_event sum_event = predefinedSumOfArray(
      estimator->sum_descriptor, kde_event);
  err = clReleaseEvent(kde_event);
  Assert(err == CL_SUCCESS);
  // Transfer the summed up contributions back, and normalize them.
  kde_float_t result;
  err = clEnqueueReadBuffer(
      ctxt->queue, estimator->result_buffer, CL_TRUE, 0,
      sizeof(kde_float_t), &result, 1, &sum_event, NULL);
  Assert(err == CL_SUCCESS);
  err = clReleaseEvent(sum_event);
  Assert(err == CL_SUCCESS);
  result *= normalization_factor / estimator->rows_in_sample;
  LOG_TIMER("Estimation");
  return result;
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
  // Finally, release the remaining buffers.
  freeEstimator(estimator);
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


// Helper function to initialize the registry.
static void ocl_initializeRegistry() {
  if (registry) return; // Don't reinitialize.
  if (IsBootstrapProcessingMode()) return; // Don't initialize during bootstrap.
  if (ocl_getContext() == NULL) return; // Don't run without OpenCl.

  // Allocate a new descriptor.
  registry = calloc(1, sizeof(ocl_estimator_registry_t));
  registry->estimator_bitmap = calloc(1, 4 * 1024 * 1024); // Enough for ~32M tables.
  registry->estimator_directory = directory_init(sizeof(Oid), 20);

  // Now open the KDE estimator table, read in all stored estimators and
  // register their descriptors.
  Relation kdeRel = heap_open(KdeModelRelationID, RowExclusiveLock);
  HeapScanDesc scan = heap_beginscan(kdeRel, SnapshotNow, 0, NULL);
  HeapTuple tuple;
  while ((tuple = heap_getnext(scan, ForwardScanDirection)) != NULL) {
    ocl_estimator_t* estimator = ocl_buildEstimatorFromCatalogEntry(
        kdeRel, tuple);
    if (estimator == NULL) continue;
    // Register the estimator.
    directory_insert(
        registry->estimator_directory, &(estimator->table), estimator);
    registry->estimator_bitmap[estimator->table / 8] |=
        (0x1 << estimator->table % 8);
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
  if (*(AttrNumber*) a > *(AttrNumber*) b) {
    return 1;
  } else if (*(AttrNumber*) a == *(AttrNumber*) b) {
    return 0;
  } else {
    return -1;
  }
}

// Helper function to print a request to stderr.
static void ocl_dumpRequest(const ocl_estimator_request_t* request) {
  unsigned int i;
  if (!request) return;
  fprintf(
      stderr, "Received estimation request for table: %i:\n",
      request->table_identifier);
  for (i = 0; i < request->range_count; ++i)
    fprintf(
        stderr, "\tColumn %i in: [%f , %f]\n", request->ranges[i].colno,
        request->ranges[i].lower_bound, request->ranges[i].upper_bound);
}

int ocl_updateRequest(
    ocl_estimator_request_t* request, AttrNumber colno,
    double* lower_bound, bool lower_included, double* upper_bound,
    bool upper_included) {
  /*
   * First, make sure to find the range entry for the given column.
   * If no column exists, insert a new one.
   */
  ocl_colrange_t* column_range = NULL;
  if (request->ranges == NULL) {
    /* We have added no ranges so far. Add this range as the first entry. */
    request->ranges = (ocl_colrange_t*) malloc(sizeof(ocl_colrange_t));
    request->range_count++;
    request->ranges->colno = colno;
    request->ranges->lower_bound = -1.0 * INFINITY;
    request->ranges->upper_bound = INFINITY;
    column_range = request->ranges;
  } else {
    /* Check whether we already have a range for this column */
    column_range = bsearch(&colno, request->ranges, request->range_count,
        sizeof(ocl_colrange_t), &compareRange);
    if (column_range == NULL) {
      /* We have to add the column. Add storage for a new value */
      request->range_count++;
      request->ranges = (ocl_colrange_t*) realloc(request->ranges,
          sizeof(ocl_colrange_t) * (request->range_count));
      /* Initialize the new column range */
      column_range = &(request->ranges[request->range_count - 1]);
      column_range->colno = colno;
      column_range->lower_bound = -1.0 * INFINITY;
      column_range->upper_bound = INFINITY;
      /* Now we have to re-sort the array */
      qsort(request->ranges, request->range_count, sizeof(ocl_colrange_t),
          &compareRange);
      /* Ok, we inserted the value. Use bsearch again to get the final position
       * of our newly inserted range. */
      column_range = bsearch(&colno, request->ranges, request->range_count,
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
  struct timeval start;
  gettimeofday(&start, NULL);
  unsigned int i;
  // Fetch the OpenCL context, initializing it if requested.
  ocl_context_t* ctxt = ocl_getContext();
  if (ctxt == NULL) return 0;
  // Make sure that the registry is initialized
  if (registry == NULL) ocl_initializeRegistry();
  // Check the registry, whether we have an estimator for the requested table.
  if (!(registry->estimator_bitmap[request->table_identifier / 8]
      & (0x1 << request->table_identifier % 8)))
    return 0;
  ocl_estimator_t* estimator = DIRECTORY_FETCH(registry->estimator_directory,
      &(request->table_identifier), ocl_estimator_t);
  if (estimator == NULL) return 0;
  // Check if the request can potentially be answered by the estimator:
  if (request->range_count > estimator->nr_of_dimensions) return 0;
  // Now check if all columns in the request are covered by the estimator:
  int request_columns = 0;
  for (i = 0; i < request->range_count; ++i) {
    request_columns |= 0x1 << request->ranges[i].colno;
  }
  if ((estimator->columns | request_columns) != estimator->columns) return 0;
  // Extract the query bounds to prepare an estimation request.
  kde_float_t* row_ranges = (kde_float_t*) malloc(
      2 * sizeof(kde_float_t) * estimator->nr_of_dimensions);
  for (i = 0; i < estimator->nr_of_dimensions; ++i) {
    row_ranges[2 * i] = -1.0 * INFINITY;
    row_ranges[2 * i + 1] = INFINITY;
  }
  for (i = 0; i < request->range_count; ++i) {
    unsigned int range_pos = estimator->column_order[request->ranges[i].colno];
    row_ranges[2 * range_pos] = request->ranges[i].lower_bound;
    row_ranges[2 * range_pos + 1] = request->ranges[i].upper_bound;
    if (request->ranges[i].lower_included) {
      row_ranges[2 * range_pos] -= 0.001;
    }
    if (request->ranges[i].upper_included) {
      row_ranges[2 * range_pos + 1] += 0.001;
    }
  }
  // Compute the selectivity.
  *selectivity = rangeKDE(ctxt, estimator, row_ranges);
  free(row_ranges);
  estimator->last_selectivity = *selectivity;
  estimator->open_estimation = true;
  // Print timing:
  if (ocl_isDebug()) {
    struct timeval now;
    gettimeofday(&now, NULL);
    long seconds = now.tv_sec - start.tv_sec;
    long useconds = now.tv_usec - start.tv_usec;
    long mtime = ((seconds) * 1000 + useconds / 1000.0) + 0.5;
    ocl_dumpRequest(request);
    fprintf(stderr, "Estimated selectivity: %f, took: %ld ms.\n", *selectivity,
        mtime);
  }
  // Schedule all steps for the online bandwidth updates.
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
  cl_int err = CL_SUCCESS;
  CREATE_TIMER(); 
  if (dimensionality > 10) {
    fprintf(stderr, "We only support models for up to 10 dimensions!\n");
    return;
  }
  // Make sure we have a context.
  ocl_context_t* ctxt = ocl_getContext();
  if (ctxt == NULL) return;
  // Make sure the registry exists.
  if (!registry) ocl_initializeRegistry();
  // Some Debug output:
  if (ocl_isDebug()) {
    fprintf(
        stderr, "Constructing an estimator for table %i.\n",
        rel->rd_node.relNode);
    fprintf(stderr, "\tColumns:");
    for (i = 0; i < dimensionality; ++i) {
      fprintf(stderr, " %i", attributes[i]);
    }
    fprintf(stderr, "\n");
    fprintf(
        stderr, "\tUsing a backing sample of %i out of %i tuples.\n",
        sample_size, rows_in_table);
  }
  // Build the column map:
  int32 column_map = 0;
  for (i = 0; i < dimensionality; ++i) {
    column_map |= 0x1 << attributes[i];
  }
  // And allocate the new estimator.
  ocl_estimator_t* estimator = allocateEstimator(
      rel->rd_node.relNode, column_map, sample_size);
  ocl_estimator_t* old_estimator = directory_insert(
      registry->estimator_directory, &(rel->rd_node.relNode), estimator);
  // If there was an existing estimator, release it.
  if (old_estimator) {
    ocl_freeEstimator(old_estimator, false);
  }
  // Register the estimator.
  registry->estimator_bitmap[rel->rd_node.relNode / 8] |= (0x1
      << rel->rd_node.relNode % 8);
  estimator->rows_in_table = rows_in_table;
  /*
   * OK, we set up the estimator. Prepare the sample for shipping it to the
   * device.
   */
  kde_float_t* host_buffer = (kde_float_t*) malloc(
      ocl_sizeOfSampleItem(estimator) * sample_size);
  for (i = 0; i < sample_size; ++i) {
    // Extract the item.
    ocl_extractSampleTuple(estimator, rel, sample[i],
        &(host_buffer[i * estimator->nr_of_dimensions]));
  }
  // Allocate a buffer of ones to initialize karma and contribution.
  kde_float_t* one_buffer = (kde_float_t*) malloc(
      sizeof(kde_float_t) * sample_size);
  // Re-scale the data to unit variance.
  for ( j = 0; j < sample_size; ++j ) {
    one_buffer[j] = 1.0f;
  }
  // Push everything to the device.
  err |= clEnqueueWriteBuffer(
      ctxt->queue, estimator->sample_buffer, CL_TRUE, 0,
      sample_size * ocl_sizeOfSampleItem(estimator), host_buffer,
      0, NULL, NULL);
  err |= clEnqueueWriteBuffer(
      ctxt->queue, estimator->sample_optimization->sample_karma_buffer,
      CL_TRUE, 0, sample_size * sizeof(kde_float_t), one_buffer,
      0, NULL, NULL);
  err |= clEnqueueWriteBuffer(
      ctxt->queue, estimator->sample_optimization->sample_contribution_buffer,
      CL_TRUE, 0, sample_size * sizeof(kde_float_t), one_buffer,
      0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  
  free(host_buffer);
  free(one_buffer);
  // Wait for the initialization to finish.
  err = clFinish(ocl_getContext()->queue);
  Assert(err == CL_SUCCESS);
  // And hand the optimization over to the model optimization.
  ocl_runModelOptimization(estimator);
  LOG_TIMER("Model Construction");
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
  if (!(registry->estimator_bitmap[relation / 8] & (0x1 << (relation % 8)))){
    return NULL;
  }
  return DIRECTORY_FETCH(
      registry->estimator_directory, &relation, ocl_estimator_t);
}

size_t ocl_sizeOfSampleItem(ocl_estimator_t* estimator) {
  return estimator->nr_of_dimensions * sizeof(kde_float_t);
}

unsigned int ocl_maxRowsInSample(ocl_estimator_t* estimator) {
  return estimator->sample_buffer_size / ocl_sizeOfSampleItem(estimator);
}

void ocl_pushEntryToSampleBufer(
    ocl_estimator_t* estimator, int position, kde_float_t* data_item) {
  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  kde_float_t one = 1.0;
  size_t transfer_size = ocl_sizeOfSampleItem(estimator);
  size_t offset = position * transfer_size;
  err |= clEnqueueWriteBuffer(
      context->queue, estimator->sample_buffer, CL_FALSE,
      offset, transfer_size, data_item, 0, NULL, NULL);
  // Initialize the metrics (both to one, so newly sampled items are not immediately replaced).
  err |= clEnqueueWriteBuffer(
      context->queue, estimator->sample_optimization->sample_karma_buffer,
      CL_FALSE, position*sizeof(kde_float_t), sizeof(kde_float_t), &one,
      0, NULL, NULL);
  err |= clEnqueueWriteBuffer(
      context->queue, estimator->sample_optimization->sample_contribution_buffer,
      CL_FALSE, position*sizeof(kde_float_t), sizeof(kde_float_t), &one,
      0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  
  err = clFinish(context->queue);
  Assert(err == CL_SUCCESS);
}

void ocl_extractSampleTuple(
    ocl_estimator_t* estimator, Relation rel,
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
    if (attribute_type == FLOAT4OID) {
      target[wpos] = DatumGetFloat4(
          heap_getattr(tuple, colno, rel->rd_att, &isNull));
    } else if (attribute_type == FLOAT8OID) {
      target[wpos] = DatumGetFloat8(
          heap_getattr(tuple, colno, rel->rd_att, &isNull));
    }
  }
}

// Helper stored procedure to import a model sample from a given file.
Datum ocl_importKDESample(PG_FUNCTION_ARGS) {
  Oid table_oid = PG_GETARG_OID(0);
  (void)table_oid;
  char *file_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
  (void)file_name;
  // Make sure that KDE is enabled.
  if (!ocl_useKDE()) {
    ereport(ERROR,
        (errcode(ERRCODE_DATATYPE_MISMATCH),
            errmsg("KDE is disabled, please set kde_enable to true!")));
    PG_RETURN_BOOL(false);
  }
  // Try to fetch the estimator:
  ocl_estimator_t* estimator = ocl_getEstimator(table_oid);
  if (estimator == NULL) {
    ereport(ERROR,
            (errcode(ERRCODE_DATATYPE_MISMATCH),
             errmsg("no KDE estimator exists for table %i", table_oid)));
    PG_RETURN_BOOL(false);
  }
  // Try to open the sample file:
  FILE* fin = fopen(file_name, "r");
  if (fin == NULL) {
    ereport(ERROR,
        (errcode(ERRCODE_DATATYPE_MISMATCH),
         errmsg("could not open file %s", file_name)));
    PG_RETURN_BOOL(false);
  }
  // Fetch the sample buffer.
  kde_float_t* sample_buffer = malloc(
      estimator->rows_in_sample * ocl_sizeOfSampleItem(estimator));

  // Read the sample file line by line:
  size_t line_buffer_size = 1024;
  char* line_buffer = malloc(line_buffer_size);
  unsigned int read_lines = 0;
  while (getline(&line_buffer, &line_buffer_size, fin) != -1) {
    if (read_lines == estimator->rows_in_sample) {
      ereport(ERROR,
              (errcode(ERRCODE_DATATYPE_MISMATCH),
               errmsg("too many tuples in sample file")));
      PG_RETURN_BOOL(false);
    }
    char* tmp = strtok(line_buffer, ",");
    int read_values = 0;
    while (tmp) {
      double value;
      sscanf(tmp, "%le", &value);
      sample_buffer[read_lines * estimator->nr_of_dimensions + read_values] = value;
      tmp = strtok(NULL, ",");
      read_values++;
    }
    read_lines++;
    if (read_values != estimator->nr_of_dimensions) {
      ereport(ERROR,
          (errcode(ERRCODE_DATATYPE_MISMATCH),
           errmsg("incorrect number of dimensions (%i) in line %i", read_values, read_lines)));
      PG_RETURN_BOOL(false);
    }
  }
  fclose(fin);
  free(line_buffer);
  if (read_lines != estimator->rows_in_sample) {
    ereport(ERROR,
            (errcode(ERRCODE_DATATYPE_MISMATCH),
             errmsg("too few tuples (%i) in sample file", read_lines)));
    PG_RETURN_BOOL(false);
  }

  // Push the new sample to the estimator.
  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  err = clEnqueueWriteBuffer(
      context->queue, estimator->sample_buffer, CL_TRUE, 0,
      estimator->rows_in_sample * ocl_sizeOfSampleItem(estimator),
      sample_buffer, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  free(sample_buffer);

  PG_RETURN_BOOL(true);
}

// Helper stored procedure to export a model sample to a given file.
Datum ocl_exportKDESample(PG_FUNCTION_ARGS) {
  Oid table_oid = PG_GETARG_OID(0);
  (void)table_oid;
  char *file_name = text_to_cstring(PG_GETARG_TEXT_PP(1));
  (void)file_name;
  // Make sure that KDE is enabled.
  if (!ocl_useKDE()) {
    ereport(ERROR,
        (errcode(ERRCODE_DATATYPE_MISMATCH),
            errmsg("KDE is disabled, please set kde_enable to true!")));
    PG_RETURN_BOOL(false);
  }
  // Try to fetch the estimator:
  ocl_estimator_t* estimator = ocl_getEstimator(table_oid);
  if (estimator == NULL) {
    ereport(ERROR,
            (errcode(ERRCODE_DATATYPE_MISMATCH),
             errmsg("no KDE estimator exists for table %i", table_oid)));
    PG_RETURN_BOOL(false);
  }
  // Try to open the file:
  FILE* fout = fopen(file_name, "w");
  if (fout == NULL) {
    ereport(ERROR,
            (errcode(ERRCODE_DATATYPE_MISMATCH),
             errmsg("could not open file %s", file_name)));
    PG_RETURN_BOOL(false);
  }
  // Fetch the sample buffer.
  kde_float_t* sample_buffer = malloc(
      estimator->rows_in_sample * ocl_sizeOfSampleItem(estimator));
  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  err = clEnqueueReadBuffer(
      context->queue, estimator->sample_buffer, CL_TRUE, 0,
      estimator->rows_in_sample * ocl_sizeOfSampleItem(estimator),
      sample_buffer, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  // Now print all samples.
  int i=0;
  for (; i<estimator->rows_in_sample; ++i) {
    // Fetch the first data item.
    kde_float_t elem = sample_buffer[i*estimator->nr_of_dimensions];
    fprintf(fout, "%le", elem);
    int j=1;
    for (; j<estimator->nr_of_dimensions; ++j) {
      elem = sample_buffer[i*estimator->nr_of_dimensions + j];
      fprintf(fout, ",%f", elem);
    }
    fprintf(fout, "\n");
  }
  fclose(fout);
  free(sample_buffer);
  PG_RETURN_BOOL(true);
}

Datum ocl_setKDEBandwidth(PG_FUNCTION_ARGS) {
  Oid table_oid = PG_GETARG_OID(0);
  ArrayType *provided_bandwidth = PG_GETARG_ARRAYTYPE_P(1);
  Oid element_type = ARR_ELEMTYPE(provided_bandwidth);
  // Make sure that KDE is enabled.
  if (!ocl_useKDE()) {
    ereport(ERROR,
        (errcode(ERRCODE_DATATYPE_MISMATCH),
            errmsg("KDE is disabled, please set kde_enable to true!")));
    PG_RETURN_BOOL(false);
  }
  // Try to fetch the estimator:
  ocl_estimator_t* estimator = ocl_getEstimator(table_oid);
  if (estimator == NULL) {
    ereport(ERROR,
            (errcode(ERRCODE_DATATYPE_MISMATCH),
             errmsg("no KDE estimator exists for table %i", table_oid)));
    PG_RETURN_BOOL(false);
  }
  // Check that the passed bandwidth is numeric:
  if (element_type != NUMERICOID) {
    ereport(ERROR,
            (errcode(ERRCODE_DATATYPE_MISMATCH),
             errmsg("bandwidth array elements must be numeric")));
    PG_RETURN_BOOL(false);
  }
  // Check that the bandwidth array has the correct size.
  int items_in_array = ArrayGetNItems(
      ARR_NDIM(provided_bandwidth), ARR_DIMS(provided_bandwidth));
  if (items_in_array != estimator->nr_of_dimensions) {
    ereport(ERROR,
            (errcode(ERRCODE_DATATYPE_MISMATCH),
             errmsg("expected bandwidth vector of size %i",
                    estimator->nr_of_dimensions)));
    PG_RETURN_BOOL(false);
  }
  // Ok, extract the bandwidths and build a buffer to store them.
  kde_float_t* new_bandwidth = malloc(
      sizeof(kde_float_t) * estimator->nr_of_dimensions);
  Datum *bandwidth_datum_array;
  deconstruct_array(
      provided_bandwidth, NUMERICOID, -1, false, 'i',
      &bandwidth_datum_array, NULL, &items_in_array);
  int i = 0;
  for (; i<items_in_array; ++i) {
    new_bandwidth[i] = DatumGetFloat8(
        DirectFunctionCall1(
            numeric_float8_no_overflow, bandwidth_datum_array[i]));
  }
  // Transfer the bandwidth to the estimator.
  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  err = clEnqueueWriteBuffer(
      context->queue, estimator->bandwidth_buffer, CL_TRUE, 0,
      sizeof(kde_float_t) * estimator->nr_of_dimensions,
      new_bandwidth, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  // We are done, clean up.
  free(new_bandwidth);
  PG_RETURN_BOOL(true);
}

Datum ocl_reoptimizeBandwidth(PG_FUNCTION_ARGS) {
  Oid table_oid = PG_GETARG_OID(0);
  // Make sure that KDE is enabled.
  if (!ocl_useKDE()) {
    ereport(ERROR,
        (errcode(ERRCODE_DATATYPE_MISMATCH),
            errmsg("KDE is disabled, please set kde_enable to true!")));
    PG_RETURN_BOOL(false);
  }
  // Try to fetch the estimator:
  ocl_estimator_t* estimator = ocl_getEstimator(table_oid);
  if (estimator == NULL) {
    ereport(ERROR,
        (errcode(ERRCODE_DATATYPE_MISMATCH),
            errmsg("no KDE estimator exists for table %i", table_oid)));
    PG_RETURN_BOOL(false);
  }
  // Alright, trigger the bandwidth reoptimization.
  ocl_runModelOptimization(estimator);
  PG_RETURN_BOOL(true);
}

Datum ocl_getBandwidth(PG_FUNCTION_ARGS) {
  Oid table_oid = PG_GETARG_OID(0);
  // Make sure that KDE is enabled.
  if (!ocl_useKDE()) {
    ereport(ERROR,
        (errcode(ERRCODE_DATATYPE_MISMATCH),
            errmsg("KDE is disabled, please set kde_enable to true!")));
    PG_RETURN_BOOL(false);
  }
  // Try to fetch the estimator:
  ocl_estimator_t* estimator = ocl_getEstimator(table_oid);
  if (estimator == NULL) {
    ereport(ERROR,
        (errcode(ERRCODE_DATATYPE_MISMATCH),
            errmsg("no KDE estimator exists for table %i", table_oid)));
    PG_RETURN_BOOL(false);
  }
  // Fetch the bandwidth array.
  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  kde_float_t* bandwidth = malloc(
      sizeof(kde_float_t) * estimator->nr_of_dimensions);
  err = clEnqueueReadBuffer(
      context->queue, estimator->bandwidth_buffer, CL_TRUE, 0,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, bandwidth,
      0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  // Now construct a
  Datum* datum_array = palloc(sizeof(Datum) * estimator->nr_of_dimensions);
  int i = 0;
  for (; i < estimator->nr_of_dimensions; ++i) {
    datum_array[i] = Float8GetDatum(bandwidth[i]);
  }
  // Clean up.
  free(bandwidth);
  PG_RETURN_ARRAYTYPE_P(
      construct_array(
          datum_array, estimator->nr_of_dimensions,
          FLOAT8OID, sizeof(double), FLOAT8PASSBYVAL, 'i'));
}

#endif /* USE_OPENCL */
