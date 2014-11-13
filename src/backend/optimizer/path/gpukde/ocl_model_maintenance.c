#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
/*
 * ocl_model_maintenance.c
 *
 *  Created on: 09.02.2014
 *      Author: mheimel
 */

#include "ocl_model_maintenance.h"

#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#include "ocl_adaptive_bandwidth.h"
#include "ocl_error_metrics.h"
#include "ocl_estimator.h"
#include "ocl_sample_maintenance.h"
#include "ocl_utilities.h"

// Optimization routines.
#include <nlopt.h>
#include "lbfgs/lbfgs.h"

#include "catalog/pg_kdefeedback.h"
#include "executor/spi.h"
#include "optimizer/path/gpukde/ocl_estimator_api.h"
#include "storage/lock.h"

// Global GUC variables
bool kde_enable_bandwidth_optimization;
int kde_bandwidth_optimization_feedback_window;

const double learning_rate = 0.01f;

// ############################################################
// # Code for adaptive bandwidth optimization (online learning).
// ############################################################

void ocl_notifyModelMaintenanceOfSelectivity(
    Oid relation, double selected, double allrows) {

  // Check if we have an estimator for this relation.
  ocl_estimator_t* estimator = ocl_getEstimator(relation);
  if (estimator == NULL) return;
  if (!estimator->open_estimation) return;  // No registered estimation.

  double selectivity = selected / allrows;
  estimator->rows_in_table = allrows;
  
  // Notify the sample maintenance of this observation so it can track the sample quality.
  ocl_notifySampleMaintenanceOfSelectivity(estimator, selectivity);

  // Update the bandwidth using online learning.
  ocl_runOnlineLearningStep(estimator, selectivity);

  // Write the error to the log file.
  ocl_reportErrorToLogFile(
      relation, estimator->last_selectivity, selectivity,
      estimator->rows_in_table);

  // We are done.
  estimator->open_estimation = false;
}

// ############################################################
// # Code for offline bandwidth optimization (batch learning).
// ############################################################

// Helper function to extract the n latest feedback records for the given
// estimator from the catalogue. The function will only return tuples that
// have feedback that matches the estimator's attributes.
//
// Returns the actual number of valid feedback records in the catalog.
static unsigned int ocl_extractLatestFeedbackRecordsFromCatalog(
    ocl_estimator_t* estimator, unsigned int requested_records,
    kde_float_t* range_buffer, kde_float_t* selectivity_buffer) {
  unsigned int current_tuple = 0;

  // Open a new scan over the feedback table.
  unsigned int i, j;
  if (SPI_connect() != SPI_OK_CONNECT) {
    fprintf(stderr, "> Error connecting to Postgres Backend.\n");
    return current_tuple;
  }
  char query_buffer[1024];
  snprintf(query_buffer, 1024,
           "SELECT columns, ranges, alltuples, qualifiedtuples FROM "
           "pg_kdefeedback WHERE \"table\" = %i ORDER BY \"timestamp\" DESC "
           "LIMIT %i;", estimator->table, requested_records);
  if (SPI_execute(query_buffer, true, 0) != SPI_OK_SELECT) {
    fprintf(stderr, "> Error querying system table.\n");
    return current_tuple;
  }
  SPITupleTable *spi_tuptable = SPI_tuptable;
  unsigned int result_tuples = SPI_processed;
  TupleDesc spi_tupdesc = spi_tuptable->tupdesc;
  for (i = 0; i < result_tuples; ++i) {
    bool isnull;
    HeapTuple record_tuple = spi_tuptable->vals[i];
    // First, check whether this record only covers columns that are part of the estimator.
    unsigned int columns_in_record = DatumGetInt32(
          SPI_getbinval(record_tuple, spi_tupdesc, 1, &isnull));
    if ((columns_in_record | estimator->columns) != estimator->columns) continue;
    // This is a valid record, initialize the range buffer.
    unsigned int pos = current_tuple * 2 * estimator->nr_of_dimensions;
    for (j=0; j<estimator->nr_of_dimensions; ++j) {
      range_buffer[pos + 2*j] = -1.0 * INFINITY;
      range_buffer[pos + 2*j + 1] = INFINITY;
    }
    // Now extract all clauses and isert them into the range buffer.
    RQClause* clauses;
    unsigned int nr_of_clauses = extract_clauses_from_buffer(DatumGetByteaP(
        SPI_getbinval(record_tuple, spi_tupdesc, 2, &isnull)), &clauses);
    for (j=0; j<nr_of_clauses; ++j) {
      // First, locate the correct column position in the estimator.
      int column_in_estimator = estimator->column_order[clauses[j].var];
      // Re-Scale the bounds, add potential padding and write them to their position.
      float8 lo = clauses[j].lobound * estimator->scale_factors[column_in_estimator];
      if (clauses[j].loinclusive != EX) lo -= 0.001;
      float8 hi = clauses[j].hibound * estimator->scale_factors[column_in_estimator];
      if (clauses[j].hiinclusive != EX) hi += 0.001;
      range_buffer[pos + 2*column_in_estimator] = lo;
      range_buffer[pos + 2*column_in_estimator + 1] = hi;
    }
    // Finally, extract the selectivity and increase the tuple count.
    double all_rows = DatumGetFloat8(
        SPI_getbinval(record_tuple, spi_tupdesc, 3, &isnull));
    double qualified_rows = DatumGetFloat8(
        SPI_getbinval(record_tuple, spi_tupdesc, 4, &isnull));
    selectivity_buffer[current_tuple++] = qualified_rows / all_rows;
  }
  // We are done :)
  SPI_finish();
  return current_tuple;
}

// Helper function that extracts feedback for the given estimator and
// pushes it to the device.
static unsigned int ocl_prepareFeedback(ocl_estimator_t* estimator,
                                        cl_mem* device_ranges,
                                        cl_mem* device_selectivities) {
  // First, we have to count how many matching feedback records are available
  // for this estimator in the query feedback table.
  if (SPI_connect() != SPI_OK_CONNECT) {
    fprintf(stderr, "> Error connecting to Postgres Backend.\n");
    return 0;
  }
  char query_buffer[1024];
  snprintf(query_buffer, 1024,
           "SELECT COUNT(*) FROM pg_kdefeedback WHERE \"table\" = %i;",
           estimator->table);
  if (SPI_execute(query_buffer, true, 0) != SPI_OK_SELECT) {
    fprintf(stderr, "> Error querying system table.\n");
    return 0;
  }
  SPITupleTable *tuptable = SPI_tuptable;
  bool isnull;
  unsigned int available_records = DatumGetInt32(
      SPI_getbinval(tuptable->vals[0], tuptable->tupdesc, 1, &isnull));
  SPI_finish();

  if (available_records == 0) {
    fprintf(stderr, "> No feedback available for table %i\n", estimator->table);
    return 0;
  }

  // Adjust the number of records according to the specified window size.
  int used_records;
  if (kde_bandwidth_optimization_feedback_window == -1)
    used_records = available_records;
  else
    used_records = Min(available_records,
                       kde_bandwidth_optimization_feedback_window);
  fprintf(stderr, "> Checking the %i latest feedback records.\n", used_records);

  // Allocate arrays and fetch the actual feedback data.
  kde_float_t* range_buffer = palloc(
      sizeof(kde_float_t) * 2 * estimator->nr_of_dimensions * used_records);
  kde_float_t* selectivity_buffer = palloc(sizeof(kde_float_t) * used_records);
  unsigned int actual_records = ocl_extractLatestFeedbackRecordsFromCatalog(
      estimator, used_records, range_buffer, selectivity_buffer);

  // Now push the records to the device to prepare the actual optimization.
  if (actual_records == 0) {
    fprintf(stderr, "> No valid feedback records found.\n");
    return 0;
  }
  fprintf(stderr, "> Found %i valid records, pushing to device.\n",
          actual_records);
  ocl_context_t* context = ocl_getContext();
  *device_ranges = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * 2 * actual_records * estimator->nr_of_dimensions,
      NULL, NULL);
  clEnqueueWriteBuffer(
      context->queue, *device_ranges, CL_FALSE, 0,
      sizeof(kde_float_t) * 2 * actual_records * estimator->nr_of_dimensions,
      range_buffer, 0, NULL, NULL);
  *device_selectivities = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * actual_records, NULL, NULL);
  clEnqueueWriteBuffer(
      context->queue, *device_selectivities, CL_FALSE, 0,
      sizeof(kde_float_t) * actual_records, selectivity_buffer, 0, NULL, NULL);
  clFinish(context->queue);
  pfree(range_buffer);
  pfree(selectivity_buffer);
  return actual_records;
}

typedef struct {
  ocl_estimator_t* estimator;
  unsigned int nr_of_observations;
  cl_mem observed_ranges;
  cl_mem observed_selectivities;
  // Temporary buffers.
  size_t stride_size;
  cl_mem gradient_accumulator_buffer;
  cl_mem error_accumulator_buffer;
  cl_mem gradient_buffer;
  cl_mem error_buffer;
} optimization_config_t;

/**
 * Counters for evaluations.
 */
int evaluations;
double start_error;
struct timeval opt_start;

/*
 * Function to compute the gradient for a penalized objective function that will
 * add a strong penalty factor to negative bandwidth values. This function is
 * passed to nlopt to compute the gradient and evaluate the function.
 */

/*
 * Callback function that computes the gradient and value for the objective
 * function at the current bandwidth.
 */
static double computeGradient(
    unsigned n, const double* bandwidth, double* gradient, void* params) {
  unsigned int i;
  cl_int err = 0;
  optimization_config_t* conf = (optimization_config_t*)params;
  ocl_estimator_t* estimator = conf->estimator;
  ocl_context_t* context = ocl_getContext();

  evaluations++;

  if (ocl_isDebug()) {
    fprintf(stderr, ">>> Evaluation %i:\n\tCurrent bandwidth:", evaluations);
    for (i=0; i<n; ++i) fprintf(stderr, " %e", bandwidth[i]);
    fprintf(stderr, "\n");
  }

  // First, transfer the current bandwidth to the device. Note that we might
  // need to cast the bandwidth to float first.
  kde_float_t* fbandwidth = NULL;
  cl_event input_transfer_event;
  if (sizeof(kde_float_t) != sizeof(double)) {
    fbandwidth = palloc(sizeof(kde_float_t) * estimator->nr_of_dimensions);
    for (i = 0; i<estimator->nr_of_dimensions; ++i) {
      fbandwidth[i] = bandwidth[i];
    }
    clEnqueueWriteBuffer(context->queue, estimator->bandwidth_buffer, CL_FALSE,
                         0, sizeof(kde_float_t) * estimator->nr_of_dimensions,
                         fbandwidth, 0, NULL, &input_transfer_event);
  } else {
    clEnqueueWriteBuffer(context->queue, estimator->bandwidth_buffer, CL_FALSE,
                         0, sizeof(kde_float_t) * estimator->nr_of_dimensions,
                         bandwidth, 0, NULL, &input_transfer_event);
  }
  // Prepare the kernel that computes a gradient for each observation.
  cl_kernel gradient_kernel = ocl_getKernel(
      ocl_getSelectedErrorMetric()->batch_kernel_name,
      estimator->nr_of_dimensions);
    // First, fix the local and global size. We identify the optimal local size
    // by looking at available and required local memory and by ensuring that
    // the local size is evenly divisible by the preferred workgroup multiple..
  size_t local_size;
  clGetKernelWorkGroupInfo(gradient_kernel, context->device,
                           CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                           &local_size, NULL);
  size_t available_local_memory;
  clGetKernelWorkGroupInfo(gradient_kernel, context->device,
                           CL_KERNEL_LOCAL_MEM_SIZE, sizeof(size_t),
                           &available_local_memory, NULL);
  available_local_memory = context->local_mem_size - available_local_memory;
  local_size = Min(
      local_size,
      available_local_memory / (3 * sizeof(kde_float_t) * estimator->nr_of_dimensions));
  size_t preferred_local_size_multiple;
  clGetKernelWorkGroupInfo(gradient_kernel, context->device,
                           CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                           sizeof(size_t), &preferred_local_size_multiple, NULL);
  local_size = preferred_local_size_multiple
      * (local_size / preferred_local_size_multiple);
  size_t global_size = local_size * (conf->nr_of_observations / local_size);
  if (global_size < conf->nr_of_observations) global_size += local_size;
    // Configure the kernel by setting all required parameters.
  err |= clSetKernelArg(gradient_kernel, 0, sizeof(cl_mem),
                        &(estimator->sample_buffer));
  err |= clSetKernelArg(gradient_kernel, 1, sizeof(unsigned int),
                        &(estimator->rows_in_sample));
  err |= clSetKernelArg(gradient_kernel, 2, sizeof(cl_mem),
                        &(conf->observed_ranges));
  err |= clSetKernelArg(gradient_kernel, 3, sizeof(cl_mem),
                        &(conf->observed_selectivities));
  err |= clSetKernelArg(gradient_kernel, 4, sizeof(unsigned int),
                        &(conf->nr_of_observations));
  err |= clSetKernelArg(gradient_kernel, 5, sizeof(cl_mem),
                        &(estimator->bandwidth_buffer));
  err |= clSetKernelArg(gradient_kernel, 6,
                        local_size * sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL);
  err |= clSetKernelArg(gradient_kernel, 7,
                        local_size * sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL);
  err |= clSetKernelArg(gradient_kernel, 8,
                        local_size * sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL);
  err |= clSetKernelArg(gradient_kernel, 9, sizeof(cl_mem),
                        &(conf->error_accumulator_buffer));
  err |= clSetKernelArg(gradient_kernel, 10, sizeof(cl_mem),
                        &(conf->gradient_accumulator_buffer));
  unsigned int stride_elements = conf->stride_size / sizeof(kde_float_t);
  err |= clSetKernelArg(gradient_kernel, 11, sizeof(unsigned int),
                        &stride_elements);
  err |= clSetKernelArg(gradient_kernel, 12, sizeof(unsigned int),
                        &(estimator->rows_in_table));
  // Compute the gradient for each observation.
  cl_event partial_gradient_event;
  err |= clEnqueueNDRangeKernel(context->queue, gradient_kernel, 1, NULL,
                                &global_size, &local_size, 1,
                                &input_transfer_event, &partial_gradient_event);
  // Sum up the individual error contributions ...
  cl_event* events = palloc(sizeof(cl_event) * (1 + estimator->nr_of_dimensions));
  events[0] = sumOfArray(
      conf->error_accumulator_buffer, conf->nr_of_observations,
      conf->error_buffer, 0, partial_gradient_event);
  // .. and the individual gradients.
  cl_mem* sub_buffers = palloc(sizeof(cl_mem) * estimator->nr_of_dimensions);
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    cl_buffer_region region;
    region.size = conf->stride_size;
    region.origin = i * conf->stride_size;
    sub_buffers[i] = clCreateSubBuffer(
        conf->gradient_accumulator_buffer, CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    events[i + 1] = sumOfArray(
        sub_buffers[i], conf->nr_of_observations,
        conf->gradient_buffer, i, partial_gradient_event);
  }
  // Now transfer the gradient back to the device.
  cl_event result_events[2];
  kde_float_t* tmp_gradient = palloc(sizeof(kde_float_t) * estimator->nr_of_dimensions);
  err |= clEnqueueReadBuffer(context->queue, conf->gradient_buffer, CL_FALSE,
                             0, sizeof(kde_float_t) * estimator->nr_of_dimensions,
                             tmp_gradient, estimator->nr_of_dimensions + 1,
                             events, &(result_events[0]));
  kde_float_t error;
  err |= clEnqueueReadBuffer(context->queue, conf->error_buffer, CL_FALSE,
                             0, sizeof(kde_float_t), &error,
                             estimator->nr_of_dimensions + 1, events,
                             &(result_events[1]));
  err |= clWaitForEvents(2, result_events);
  
  if (err != 0) {
    fprintf(stderr, "OpenCL functions failed to compute gradient.\n");
  }
  
  error /= conf->nr_of_observations;
  if (evaluations == 1) start_error = error;
  struct timeval now; gettimeofday(&now, NULL);
  long seconds = now.tv_sec - opt_start.tv_sec;
  long useconds = now.tv_usec - opt_start.tv_usec;
  long mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
  // Ok, cool. Transfer the bandwidth back.
  fprintf(stderr, "\r\tOptimization round %i. Current error: %f "
          "(started at %f), took: %ld ms.",
          evaluations, error, start_error, mtime);
  // Finally, cast back to double.
  if (gradient) {
    for (i = 0; i<estimator->nr_of_dimensions; ++i) {
    // Apply the gradient normalization.
      double h = bandwidth[i];
      gradient[i] = tmp_gradient[i] * M_SQRT2 / (
          sqrt(M_PI) * h * h * pow(2.0, estimator->nr_of_dimensions) *
          conf->nr_of_observations * estimator->rows_in_sample);
    }
    if (ocl_isDebug()) {
      fprintf(stderr, "\n\tGradient:");
      for (i=0; i<n; ++i) fprintf(stderr, " %e", gradient[i]);
      fprintf(stderr, "\n");
    }
  }
  // Ok, clean everything up.
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    clReleaseMemObject(sub_buffers[i]);
    clReleaseEvent(events[i]);
  }
  pfree(tmp_gradient);
  if (fbandwidth) pfree(fbandwidth);
  pfree(events);
  pfree(sub_buffers);
  clReleaseEvent(input_transfer_event);
  clReleaseEvent(partial_gradient_event);
  clReleaseEvent(result_events[0]);
  clReleaseEvent(result_events[1]);
  clReleaseKernel(gradient_kernel);
  return error;
}

static void ocl_setScottsBandwidth(ocl_estimator_t* estimator) {
  ocl_context_t* context = ocl_getContext();
  // First, we need to compute the variance for each dimension.
  unsigned int i;
  kde_float_t* variances = malloc(
      sizeof(kde_float_t) * estimator->nr_of_dimensions);
  cl_mem* buffers = malloc(sizeof(cl_mem) * estimator->nr_of_dimensions);
  cl_mem averages = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);
  cl_event* events = malloc(sizeof(cl_event) * estimator->nr_of_dimensions);
  size_t sample_size = estimator->rows_in_sample;
  size_t dimensions = estimator->nr_of_dimensions;
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    // Allocate all required buffers.
    buffers[i] = clCreateBuffer(
        context->context, CL_MEM_READ_WRITE,
        sizeof(kde_float_t) * estimator->rows_in_sample, NULL, NULL);
    // First we extract all sample components for this dimension.
    cl_kernel extractComponents = ocl_getKernel("extract_dimension", 0);
    clSetKernelArg(
        extractComponents, 0, sizeof(cl_mem), &(estimator->sample_buffer));
    clSetKernelArg(extractComponents, 1, sizeof(cl_mem), &(buffers[i]));
    clSetKernelArg(extractComponents, 2, sizeof(unsigned int), &i);
    clSetKernelArg(extractComponents, 3, sizeof(unsigned int),
        &(estimator->nr_of_dimensions));
    cl_event extraction_event;
    clEnqueueNDRangeKernel(
        context->queue, extractComponents, 1, NULL, &sample_size, NULL,
        0, NULL, &extraction_event);
    // Now we sum them up, so we can compute the average.
    cl_event average_summation_event = sumOfArray(
        buffers[i], estimator->rows_in_sample, averages, i, extraction_event);
    // Alright, we can compute the variance contributions from each point.
    cl_kernel precomputeVariance = ocl_getKernel("precompute_variance", 0);
    clSetKernelArg(precomputeVariance, 0, sizeof(cl_mem), &(buffers[i]));
    clSetKernelArg(precomputeVariance, 1, sizeof(cl_mem), &averages);
    clSetKernelArg(precomputeVariance, 2, sizeof(unsigned int), &i);
    clSetKernelArg(precomputeVariance, 3, sizeof(unsigned int),
        &(estimator->rows_in_sample));
    cl_event variance_event;
    clEnqueueNDRangeKernel(
        context->queue, precomputeVariance, 1, NULL, &sample_size, NULL,
        1, &average_summation_event, &variance_event);
    // We now sum up the single contributions to compute the variance.
    cl_event variance_summation_event = sumOfArray(
        buffers[i], estimator->rows_in_sample, averages, i, variance_event);
    // Finally, we can compute and store the bandwidth for this value.
    cl_kernel finalizeBandwidth = ocl_getKernel("set_scotts_bandwidth", 0);
    clSetKernelArg(finalizeBandwidth, 0, sizeof(cl_mem), &averages);
    clSetKernelArg(finalizeBandwidth, 1, sizeof(cl_mem),
        &(estimator->bandwidth_buffer));
    clSetKernelArg(finalizeBandwidth, 2, sizeof(unsigned int), &i);
    clSetKernelArg(finalizeBandwidth, 3, sizeof(unsigned int),
        &(estimator->nr_of_dimensions));
    clSetKernelArg(finalizeBandwidth, 4, sizeof(unsigned int),
        &(estimator->rows_in_sample));
    clEnqueueNDRangeKernel(
        context->queue, finalizeBandwidth, 1, NULL, &dimensions, NULL,
        1, &variance_summation_event, &events[i]);
    // Clean up.
    clReleaseMemObject(buffers[i]);
    clReleaseEvent(extraction_event);
    clReleaseEvent(average_summation_event);
    clReleaseEvent(variance_event);
    clReleaseEvent(variance_summation_event);
  }
  clReleaseMemObject(averages);
  // Wait for the events to finalize:
  clWaitForEvents(estimator->nr_of_dimensions, events);
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    clReleaseEvent(events[i]);
  }
  // Clean up.
  free(variances);
  free(events);
}

void ocl_runModelOptimization(ocl_estimator_t* estimator) {
  if (estimator == NULL) return;
  // Set the rule-of-thumb bandwidth to initialize the estimator.
  ocl_setScottsBandwidth(estimator);
  // Now check if we do a full bandwidth optimization.
  if (!kde_enable_bandwidth_optimization) return;
  if (ocl_isDebug()) {
    fprintf(
        stderr, "Beginning model optimization for estimator on table %i\n",
        estimator->table);
  }
  // First, we need to fetch the feedback records for this table and push them
  // to the device.
  cl_mem device_ranges, device_selectivites;
  unsigned int feedback_records = ocl_prepareFeedback(
      estimator, &device_ranges, &device_selectivites);
  if (feedback_records == 0) return;

  // We need to transfer the bandwidth to the host.
  kde_float_t* fbandwidth = palloc(
      sizeof(kde_float_t) * estimator->nr_of_dimensions);
  ocl_context_t* context = ocl_getContext();
  clEnqueueReadBuffer(
      context->queue, estimator->bandwidth_buffer, CL_TRUE, 0,
      sizeof(kde_float_t) * estimator->nr_of_dimensions,
      fbandwidth, 0, NULL, NULL);
  // Cast to double (nlopt operates on double).
  double* bandwidth = lbfgs_malloc(estimator->nr_of_dimensions);
  unsigned int i;
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    bandwidth[i] = fbandwidth[i];
  }

  // Package all required buffers.
  optimization_config_t params;
  params.estimator = estimator;
  params.nr_of_observations = feedback_records;
  params.observed_ranges = device_ranges;
  params.observed_selectivities = device_selectivites;
  params.error_accumulator_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * feedback_records, NULL, NULL);
  // Allocate a buffer to hold temporary gradient contributions. This buffer
  // will keep D contributions per observation. We store all contributions
  // consecutively (i.e. 111222333444). For optimal performance, we therefore
  // have to make sure that the consecutive regions (strides) have a size that
  // is aligned to the required machine alignment.
  params.stride_size = sizeof(kde_float_t) * feedback_records;
  if ((params.stride_size * 8) % context->required_mem_alignment) {
    // The stride size is misaligned, add some padding.
    params.stride_size *= 8;
    params.stride_size =
        (1 + params.stride_size / context->required_mem_alignment)
        * context->required_mem_alignment;
    params.stride_size /= 8;
  }
  params.gradient_accumulator_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      estimator->nr_of_dimensions * params.stride_size,
      NULL, NULL);
  params.gradient_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      estimator->nr_of_dimensions * sizeof(kde_float_t), NULL, NULL);
  params.error_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE, sizeof(kde_float_t), NULL, NULL);
  // Ok, we are prepared. Call the optimization routine.
  gettimeofday(&opt_start, NULL);
  evaluations = 0;
  fprintf(stderr, "> Starting numerical optimization of the model:\n");
  // Prepare the bound constraints.
  double* lower_bounds = palloc(sizeof(double) * estimator->nr_of_dimensions);
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    lower_bounds[i] = 1e-5;    // We never want to be negative.
  }
  // We use 10x the heuristic bandwidth as our upper bound:
  double* upper_bounds = palloc(sizeof(double) * estimator->nr_of_dimensions);
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    upper_bounds[i] = 1e3;
  }
  // Create the optimization parameter.
  nlopt_opt global_optimizer= nlopt_create(
      NLOPT_LD_MMA, estimator->nr_of_dimensions);
  nlopt_set_lower_bounds(global_optimizer, lower_bounds);
  nlopt_set_upper_bounds(global_optimizer, upper_bounds);
  nlopt_set_min_objective(global_optimizer, computeGradient, &params);
  nlopt_set_ftol_abs(global_optimizer, 1e-10);
  nlopt_set_maxeval(global_optimizer, 100);
  double tmp;
  int err = nlopt_optimize(global_optimizer, bandwidth, &tmp);
  if (ocl_isDebug()) {
    if (err < 0) {
      fprintf(stderr, "\nOptimization failed: %i!", err);
    } else {
      fprintf(stderr, "\nNew bandwidth:");
      for ( i = 0; i < estimator->nr_of_dimensions ; ++i)
        fprintf(stderr, " %e", bandwidth[i]);
      fprintf(stderr, "\n");
    }
  }
  nlopt_destroy(global_optimizer);
  // Transfer the bandwidth to the device.
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    fbandwidth[i] = bandwidth[i];
  }
  clEnqueueWriteBuffer(
      context->queue, estimator->bandwidth_buffer, CL_TRUE, 0,
      sizeof(kde_float_t) * estimator->nr_of_dimensions,
      fbandwidth, 0, NULL, NULL);
  // Clean up.
  pfree(fbandwidth);
  lbfgs_free(bandwidth);
  clReleaseMemObject(params.error_buffer);
  clReleaseMemObject(params.gradient_buffer);
  clReleaseMemObject(params.error_accumulator_buffer);
  clReleaseMemObject(params.gradient_accumulator_buffer);
  clReleaseMemObject(device_ranges);
  clReleaseMemObject(device_selectivites);
}
