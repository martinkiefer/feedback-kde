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

#include "ocl_estimator.h"
#include "ocl_sample_maintenance.h"

#include "lbfgs/lbfgs.h"

#include "catalog/pg_kdefeedback.h"
#include "executor/spi.h"
#include "storage/lock.h"

// Global GUC variables
bool kde_enable_bandwidth_optimization;
int kde_bandwidth_optimization_feedback_window;
bool kde_enable_adaptive_bandwidth;
int kde_adaptive_bandwidth_minibatch_size;
char* kde_estimation_quality_logfile_name;
int kde_error_metric;


// ############################################################
// # Define estimation error metrics.
// ############################################################
typedef struct error_metric {
	const char* name;
	float (*function)(float, float);
	float (*gradient_factor)(float, float);
	const char* batch_kernel_name;
} error_metric_t;


// Functions that implement the error metrics.
static float QuadraticError(float actual, float expected) {
  return (actual - expected) * (actual - expected);
}
static float QuadraticErrorGradientFactor(float actual, float expected) {
  return 2 * (actual - expected);
}
static float QErrror(float actual, float expected) {
  // Constants are required to avoid computing the log of 0.
  float tmp = log(0.0001f + actual) - log(0.0001f + expected);
  return tmp * tmp;
}
static float QErrorGradientFactor(float actual, float expected) {
  return 2 * (log(0.0001f + actual) - log(0.0001f + expected)) / (0.0001f + actual);
}
static float AbsoluteError(float actual, float expected) {
  return fabs(actual - expected);
}
static float AbsoluteErrorGradientFactor(float actual, float expected) {
  if (actual > expected) {
    return 1;
  } else {
    return -1;
  }
}
static float RelativeError(float actual, float expected) {
  // Not entirely correct, but robust against zero estimates.
  return fabs(actual - expected) / (0.0001f + expected);
}
static float RelativeErrorGradientFactor(float actual, float expected) {
  if (actual > expected) {
    return 1.0f / (0.0001f + expected);
  } else {
    return -1.0f / (0.0001f + expected);
  }
}

// Array of all available metrics.
static error_metric_t error_metrics[] = {
   {
      "Absolute",  &AbsoluteError, &AbsoluteErrorGradientFactor,
      "computeBatchGradientAbsolute"
   },
   {
      "Relative", &RelativeError, &RelativeErrorGradientFactor,
      "computeBatchGradientRelative"
   },
   {
      "Quadratic", &QuadraticError, &QuadraticErrorGradientFactor,
      "computeBatchGradientQuadratic"
   },
   {
      "QError", &QErrror, &QErrorGradientFactor, "computeBatchGradientQ"
   }
};

// ############################################################
// # Code for estimation error reporting.
// ############################################################

static FILE* estimation_quality_log_file = NULL;

void assign_kde_estimation_quality_logfile_name(const char *newval, void *extra) {
  if (estimation_quality_log_file != NULL) fclose(estimation_quality_log_file);
  estimation_quality_log_file = fopen(newval, "w");
  if (estimation_quality_log_file == NULL) return;
  // Write a header to the file to specify all registered error metrics.
  unsigned int i;
  fprintf(estimation_quality_log_file, "Relation ID");
  for (i=0; i<sizeof(error_metrics)/sizeof(error_metric_t); ++i) {
	  fprintf(estimation_quality_log_file, " ; %s", error_metrics[i].name);
  }
  fprintf(estimation_quality_log_file, "\n");
  fflush(estimation_quality_log_file);
}

static void ocl_reportErrorToLogFile(Oid relation, float actual, float expected) {
  if (estimation_quality_log_file == NULL) return;
  // Compute the estimation error for all metrics and write them to the file.
  unsigned int i;
  fprintf(estimation_quality_log_file, "%u", relation);
  for (i=0; i<sizeof(error_metrics)/sizeof(error_metric_t); ++i) {
 	  float error = (*(error_metrics[i].function))(actual, expected);
 	  fprintf(estimation_quality_log_file, " ; %.3f", error);
   }
   fprintf(estimation_quality_log_file, "\n");
   fflush(estimation_quality_log_file);
}

/**
 * Helper function to efficiently compute the sum of an input buffer.
 *
 * The computed sum is written to the offset result_buffer_offset in the
 * specified result_buffer.
 *
 * The function returns an event to wait for the computation to complete.
 */
static cl_event sumOfArray(cl_mem input_buffer, unsigned int elements,
                           cl_mem result_buffer, unsigned int result_buffer_offset,
                           cl_event external_event) {
  cl_int err = 0;
  ocl_context_t* context = ocl_getContext();
  cl_event init_event;
  cl_event events[] = { NULL, NULL };
  unsigned int nr_of_events = 0;
  // Fetch the required sum kernels:
  cl_kernel init_buffer = ocl_getKernel("init_zero", 0);
  cl_kernel fast_sum = ocl_getKernel("sum_par", 0);
  cl_kernel slow_sum = ocl_getKernel("sum_seq", 0);
  struct timeval start; gettimeofday(&start, NULL);
  // Determine the kernel parameters:
  size_t global_size = 0;
  size_t local_size = context->max_workgroup_size;
  size_t processors = context->max_compute_units;
  // Figure out how many elements we can aggregate per thread in the parallel part:
  unsigned int tuples_per_thread = elements / (processors * local_size);
  // Now compute the configuration of the sequential kernel:
  unsigned int slow_kernel_data_offset = processors * tuples_per_thread * local_size;
  unsigned int slow_kernel_elements = elements - slow_kernel_data_offset;
  unsigned int slow_kernel_result_offset = processors;
  // Allocate a temporary result buffer.
  cl_mem tmp_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(float) * (processors + 1), NULL, NULL);
  err |= clSetKernelArg(init_buffer, 0, sizeof(cl_mem), &tmp_buffer);
  global_size = processors + 1;
  err |= clEnqueueNDRangeKernel(
      context->queue, init_buffer, 1, NULL, &global_size,
      NULL, 1, &external_event, &init_event);
  // Ok, we selected the correct kernel and parameters. Now prepare the arguments.
  err |= clSetKernelArg(fast_sum, 0, sizeof(cl_mem), &input_buffer);
  err |= clSetKernelArg(fast_sum, 1, sizeof(cl_mem), &tmp_buffer);
  err |= clSetKernelArg(fast_sum, 2, sizeof(unsigned int), &tuples_per_thread);
  err |= clSetKernelArg(slow_sum, 0, sizeof(cl_mem), &input_buffer);
  err |= clSetKernelArg(slow_sum, 1, sizeof(unsigned int), &slow_kernel_data_offset);
  err |= clSetKernelArg(slow_sum, 2, sizeof(unsigned int), &slow_kernel_elements);
  err |= clSetKernelArg(slow_sum, 3, sizeof(cl_mem), &tmp_buffer);
  err |= clSetKernelArg(slow_sum, 4, sizeof(unsigned int), &slow_kernel_result_offset);
  // Fire the kernel
  if (tuples_per_thread) {
    global_size = local_size * processors;
    cl_event event;
    err |= clEnqueueNDRangeKernel(
        context->queue, fast_sum, 1, NULL, &global_size,
        &local_size, 1, &init_event, &event);
    events[nr_of_events++] = event;
  }
  if (slow_kernel_elements) {
    cl_event event;
    err |= clEnqueueTask(context->queue, slow_sum, 1, &init_event, &event);
    events[nr_of_events++] = event;
  }
  // Now perform a final pass over the data to compute the aggregate.
  slow_kernel_data_offset = 0;
  slow_kernel_elements = processors + 1;
  slow_kernel_result_offset = 0;
  err |= clSetKernelArg(slow_sum, 0, sizeof(cl_mem), &tmp_buffer);
  err |= clSetKernelArg(slow_sum, 1, sizeof(unsigned int), &slow_kernel_data_offset);
  err |= clSetKernelArg(slow_sum, 2, sizeof(unsigned int), &slow_kernel_elements);
  err |= clSetKernelArg(slow_sum, 3, sizeof(cl_mem), &result_buffer);
  err |= clSetKernelArg(slow_sum, 4, sizeof(unsigned int), &result_buffer_offset);
  cl_event finalize_event;
  err |= clEnqueueTask(context->queue, slow_sum, nr_of_events, events, &finalize_event);

  // Clean up ...
  clReleaseKernel(slow_sum);
  clReleaseKernel(fast_sum);
  clReleaseKernel(init_buffer);
  clReleaseMemObject(tmp_buffer);
  clReleaseEvent(init_event);
  if (events[0]) clReleaseEvent(events[0]);
  if (events[1]) clReleaseEvent(events[1]);

  return finalize_event;
}

const float learning_rate = 0.05f;

// ############################################################
// # Code for adaptive bandwidth optimization (online learning).
// ############################################################

/**
 * Helper function to compute a single online learning step.
 */
static void ocl_runOnlineLearningStep(ocl_estimator_t* estimator,
                                      float selectivity) {
  if (!kde_enable_adaptive_bandwidth) return;

  cl_int err;
  ocl_context_t* context = ocl_getContext();

  // Check whether the estimator already has a designated buffer to accumulate gradients.
  if (estimator->gradient_accumulator == NULL) {
    estimator->gradient_accumulator = clCreateBuffer(
        context->context, CL_MEM_READ_WRITE,
        sizeof(float) * estimator->nr_of_dimensions, NULL, NULL);
  }

  struct timeval start; gettimeofday(&start, NULL);

  // Allocate a buffer to hold the temporary gradient contributions.
  size_t stride_size = sizeof(float) * estimator->rows_in_sample;
  if ((stride_size * 8) % context->required_mem_alignment) {
    // The stride size is misaligned, add some padding.
    stride_size *= 8;
    stride_size = (1 + stride_size / context->required_mem_alignment)
                * context->required_mem_alignment;
    stride_size /= 8;
  }
  unsigned int result_stride_elements = stride_size / sizeof(float);

  cl_mem gradient_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      stride_size * estimator->nr_of_dimensions, NULL, &err);
  if (gradient_buffer == NULL || err != CL_SUCCESS) {
    fprintf(stderr, "Could not allocate buffer for gradient computation.\n");
    return;
  }

  // Now calculate the gradient contributions from each sample point.
  cl_kernel computeSingleGradient = ocl_getKernel("computeSingleGradient",
                                                  estimator->nr_of_dimensions);
  // Figure out the maximum local size this kernel supports.
  size_t local_size;
  clGetKernelWorkGroupInfo(computeSingleGradient, context->device,
                           CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                           &local_size, NULL);
  // Now cap this to the local memory size requirements.
  size_t available_local_memory;
  clGetKernelWorkGroupInfo(computeSingleGradient, context->device,
                           CL_KERNEL_LOCAL_MEM_SIZE, sizeof(size_t),
                           &available_local_memory, NULL);
  available_local_memory = context->local_mem_size - available_local_memory;
  local_size = Min(
      local_size,
      available_local_memory / (sizeof(float) * estimator->nr_of_dimensions));
  // Finally, cap the local size to a multiple of the preferred multiple size.
  size_t preferred_local_size_multiple;
  clGetKernelWorkGroupInfo(computeSingleGradient, context->device,
                           CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                           sizeof(size_t), &preferred_local_size_multiple, NULL);
  local_size = preferred_local_size_multiple
             * (local_size / preferred_local_size_multiple);
  // Ok, compute the global size to match this local size.
  size_t global_size = local_size * (estimator->rows_in_sample / local_size);
  if (global_size < estimator->rows_in_sample) global_size+=local_size;

  err |= clSetKernelArg(computeSingleGradient, 0, sizeof(cl_mem),
                        &(estimator->sample_buffer));
  err |= clSetKernelArg(computeSingleGradient, 1, sizeof(unsigned int),
                        &(estimator->rows_in_sample));
  err |= clSetKernelArg(computeSingleGradient, 2, sizeof(cl_mem),
                        &(context->input_buffer));
  err |= clSetKernelArg(computeSingleGradient, 3, sizeof(cl_mem),
                        &(estimator->bandwidth_buffer));
  err |= clSetKernelArg(computeSingleGradient, 4, available_local_memory, NULL);
  err |= clSetKernelArg(computeSingleGradient, 5, sizeof(cl_mem),
                        &gradient_buffer);
  err |= clSetKernelArg(computeSingleGradient, 6, sizeof(unsigned int),
                        &result_stride_elements);
  cl_event gradient_event;
  err |= clEnqueueNDRangeKernel(context->queue, computeSingleGradient, 1, NULL,
                                &global_size, &local_size, 0, NULL,
                                &gradient_event);

  // Ok cool, now compute the actual gradient by summing up the individual contributions.
  cl_mem gradient = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(float) * estimator->nr_of_dimensions, NULL, &err);
  cl_event* events = palloc(sizeof(cl_event) * estimator->nr_of_dimensions);
  cl_mem* sub_buffers = palloc(sizeof(cl_mem) * estimator->nr_of_dimensions);
  unsigned int i;
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    cl_buffer_region region;
    region.size = stride_size;
    region.origin = i * stride_size;
    sub_buffers[i] = clCreateSubBuffer(
        gradient_buffer, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
        &region, &err);
    events[i] = sumOfArray(sub_buffers[i], estimator->rows_in_sample,
                           gradient, i, gradient_event);
  }
  // Finally, compute the gradient scaling factor.
  float scale_factor = 1.0f
      / (pow(2.0f, estimator->nr_of_dimensions) * estimator->rows_in_sample);
  scale_factor *= (*(error_metrics[kde_error_metric].gradient_factor))(
      estimator->last_selectivity, selectivity);

  // Now accumulate the bandwidth.
  cl_kernel applyGradient = ocl_getKernel("applyGradient", 0);
  bool cap_to_positive;
  if (kde_adaptive_bandwidth_minibatch_size == 1) {
    // We use online learning, directly apply the gradient to the bandwidth.
    err |= clSetKernelArg(applyGradient, 0, sizeof(cl_mem),
                          &(estimator->bandwidth_buffer));
    scale_factor *= learning_rate;
    cap_to_positive = true; // Never allow negative bandwidths.
  } else {
    // We use mini-batch learning, accumulate the gradient.
    err |= clSetKernelArg(applyGradient, 0, sizeof(float),
                          &(estimator->gradient_accumulator));
    cap_to_positive = false;  // But allow negative gradients.
  }
  err |= clSetKernelArg(applyGradient, 1, sizeof(cl_mem), &gradient);
  err |= clSetKernelArg(applyGradient, 2, sizeof(bool), &cap_to_positive);
  err |= clSetKernelArg(applyGradient, 3, sizeof(float), &scale_factor);
  global_size = estimator->nr_of_dimensions;
  cl_event apply_event;
  err |= clEnqueueNDRangeKernel(context->queue, applyGradient, 1, NULL,
                                &global_size, NULL, estimator->nr_of_dimensions,
                                events, &apply_event);

  // If we use mini-batch learning, we may need to apply the accumulated gradient to the bandwidth.
  if (kde_adaptive_bandwidth_minibatch_size > 1) {
    estimator->nr_of_accumulated_gradients++;
    if (estimator->nr_of_accumulated_gradients >= kde_adaptive_bandwidth_minibatch_size) {
      // We have filled a mini-batch, apply the gradient.
      scale_factor = learning_rate / estimator->nr_of_accumulated_gradients;
      cap_to_positive = true; // Never allow negative bandwidths.
      err |= clSetKernelArg(applyGradient, 0, sizeof(cl_mem),
                            &(estimator->bandwidth_buffer));
      err |= clSetKernelArg(applyGradient, 1, sizeof(cl_mem),
                            &(estimator->gradient_accumulator));
      err |= clSetKernelArg(applyGradient, 2, sizeof(bool), &cap_to_positive);
      err |= clSetKernelArg(applyGradient, 3, sizeof(float), &scale_factor);
      cl_event mini_batch_apply_event;
      err |= clEnqueueNDRangeKernel(context->queue, applyGradient, 1, NULL,
                                    &global_size, NULL, 1, &apply_event,
                                    &mini_batch_apply_event);
      // Zero out the accumulation buffer.
      cl_kernel init_zero = ocl_getKernel("init_zero", 0);
      err |= clSetKernelArg(init_zero, 0, sizeof(cl_mem),
                            &(estimator->gradient_accumulator));
      err |= clEnqueueNDRangeKernel(context->queue, init_zero, 1, NULL,
                                    &global_size, NULL, 1,
                                    &mini_batch_apply_event, NULL);
      clReleaseEvent(mini_batch_apply_event);
      clReleaseKernel(init_zero);
    }
  }

  // Clean up.
  clFinish(context->queue);
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    clReleaseMemObject(sub_buffers[i]);
    clReleaseEvent(events[i]);
  }
  clReleaseMemObject(gradient_buffer);
  clReleaseEvent(gradient_event);
  clReleaseEvent(apply_event);
  pfree(events);
  pfree(sub_buffers);

  // Print timing:
  struct timeval now; gettimeofday(&now, NULL);
  long seconds = now.tv_sec - start.tv_sec;
  long useconds = now.tv_usec - start.tv_usec;
  long mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
  fprintf(stderr, "Adjusted bandwidth, took: %ld ms.\n", mtime);
}


void ocl_notifyModelMaintenanceOfSelectivity(Oid relation, float selectivity) {
  // Check if we have an estimator for this relation.
  ocl_estimator_t* estimator = ocl_getEstimator(relation);
  if (estimator == NULL) return;
  if (!estimator->open_estimation) return;  // No registered estimation.

  // Notify the sample maintenance of this observation so it can track the sample quality.
  //ocl_notifySampleMaintenanceOfSelectivity(estimator, selectivity);

  // Update the bandwidth using online learning.
  ocl_runOnlineLearningStep(estimator, selectivity);

  // Write the error to the log file.
  ocl_reportErrorToLogFile(relation, estimator->last_selectivity, selectivity);

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
    float* range_buffer, float* selectivity_buffer) {
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
      range_buffer[pos + 2*j] = -1.0f * INFINITY;
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
      float lo = clauses[j].lobound / estimator->scale_factors[column_in_estimator];
      if (clauses[j].loinclusive != EX) lo -= 0.001f;
      float hi = clauses[j].hibound / estimator->scale_factors[column_in_estimator];
      if (clauses[j].hiinclusive != EX) hi += 0.001f;
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
  float* range_buffer = palloc(
      sizeof(float) * 2 * estimator->nr_of_dimensions * used_records);
  float* selectivity_buffer = palloc(sizeof(float) * used_records);
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
      sizeof(float) * 2 * actual_records * estimator->nr_of_dimensions,
      NULL, NULL);
  clEnqueueWriteBuffer(
      context->queue, *device_ranges, CL_FALSE, 0,
      sizeof(float) * 2 * actual_records * estimator->nr_of_dimensions,
      range_buffer, 0, NULL, NULL);
  *device_selectivities = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(float) * actual_records, NULL, NULL);
  clEnqueueWriteBuffer(
      context->queue, *device_selectivities, CL_FALSE, 0,
      sizeof(float) * actual_records, selectivity_buffer, 0, NULL, NULL);
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
} optimization_payload_t;

int evaluations;

// Function to compute the gradient with regard to the current bandwidth
// parameters and the provided observations.
static lbfgsfloatval_t computeGradient(
    void* user_params,
    const lbfgsfloatval_t* current_bandwidth,
    lbfgsfloatval_t* gradient,
    const int n,
    const lbfgsfloatval_t step
    ) {
  unsigned int i;
  cl_int err = 0;
  optimization_payload_t* params = (optimization_payload_t*)user_params;
  ocl_estimator_t* estimator = params->estimator;
  ocl_context_t* context = ocl_getContext();

  evaluations++;

  fprintf(stderr, "\t>>>> Iteration %i, bandwidth:", evaluations);
  for (i = 0; i<estimator->nr_of_dimensions; ++i) {
    gradient[i] /= params->nr_of_observations;
    fprintf(stderr, " %f", current_bandwidth[i]);
  }
  fprintf(stderr, "\n");

  // First, we need to push the new bandwidth to the device.
  cl_event input_transfer_event;
  clEnqueueWriteBuffer(context->queue, estimator->bandwidth_buffer, CL_FALSE,
                       0, sizeof(float) * estimator->nr_of_dimensions,
                       current_bandwidth, 0, NULL, &input_transfer_event);
  // Now compute the gradient contributions.
  cl_kernel gradient_kernel = ocl_getKernel(
      error_metrics[kde_error_metric].batch_kernel_name,
      estimator->nr_of_dimensions);
  // Determine the maximum local workgroup size for this kernel.
  size_t local_size;
  clGetKernelWorkGroupInfo(gradient_kernel, context->device,
                           CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                           &local_size, NULL);
  // Now cap this to the local memory size requirements.
  size_t available_local_memory;
  clGetKernelWorkGroupInfo(gradient_kernel, context->device,
                           CL_KERNEL_LOCAL_MEM_SIZE, sizeof(size_t),
                           &available_local_memory, NULL);
  available_local_memory = context->local_mem_size - available_local_memory;
  local_size = Min(
      local_size,
      available_local_memory / (3 * sizeof(float) * estimator->nr_of_dimensions));
  // Finally, cap the local size to a multiple of the preferred multiple size.
  size_t preferred_local_size_multiple;
  clGetKernelWorkGroupInfo(gradient_kernel, context->device,
                           CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                           sizeof(size_t), &preferred_local_size_multiple, NULL);
  local_size = preferred_local_size_multiple
             * (local_size / preferred_local_size_multiple);
  // Ok, compute the global size to match this local size.
  size_t global_size = local_size * (params->nr_of_observations / local_size);
  if (global_size < params->nr_of_observations) global_size += local_size;
  // Cool, parameterize the kernel.
  err |= clSetKernelArg(gradient_kernel, 0, sizeof(cl_mem),
                        &(estimator->sample_buffer));
  err |= clSetKernelArg(gradient_kernel, 1, sizeof(unsigned int),
                        &(estimator->rows_in_sample));
  err |= clSetKernelArg(gradient_kernel, 2, sizeof(cl_mem),
                        &(params->observed_ranges));
  err |= clSetKernelArg(gradient_kernel, 3, sizeof(cl_mem),
                        &(params->observed_selectivities));
  err |= clSetKernelArg(gradient_kernel, 4, sizeof(unsigned int),
                        &(params->nr_of_observations));
  err |= clSetKernelArg(gradient_kernel, 5, sizeof(cl_mem),
                        &(estimator->bandwidth_buffer));
  err |= clSetKernelArg(gradient_kernel, 6,
                        sizeof(float) * estimator->nr_of_dimensions, NULL);
  err |= clSetKernelArg(gradient_kernel, 7,
                        sizeof(float) * estimator->nr_of_dimensions, NULL);
  err |= clSetKernelArg(gradient_kernel, 8,
                        sizeof(float) * estimator->nr_of_dimensions, NULL);
  err |= clSetKernelArg(gradient_kernel, 9, sizeof(cl_mem),
                        &(params->error_accumulator_buffer));
  err |= clSetKernelArg(gradient_kernel, 10, sizeof(cl_mem),
                        &(params->gradient_accumulator_buffer));
  unsigned int stride_elements = params->stride_size / sizeof(float);
  err |= clSetKernelArg(gradient_kernel, 11, sizeof(unsigned int),
                        &stride_elements);
  cl_event partial_gradient_event;
  err |= clEnqueueNDRangeKernel(context->queue, gradient_kernel, 1, NULL,
                                &global_size, &local_size, 1,
                                &input_transfer_event, &partial_gradient_event);
  cl_event* events = palloc(sizeof(cl_event) * (1 + estimator->nr_of_dimensions));
  cl_mem* sub_buffers = palloc(sizeof(cl_mem) * estimator->nr_of_dimensions);
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    cl_buffer_region region;
    region.size = params->stride_size;
    region.origin = i * params->stride_size;
    sub_buffers[i] = clCreateSubBuffer(
        params->gradient_accumulator_buffer, CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    events[i + 1] = sumOfArray(sub_buffers[i], params->nr_of_observations,
                           params->gradient_buffer, i, partial_gradient_event);
  }
  events[0] = sumOfArray(
      params->error_accumulator_buffer, params->nr_of_observations,
      params->error_buffer, 0, partial_gradient_event);
  // Transfer the gradient and error back.
  cl_event result_events[2];
  err |= clEnqueueReadBuffer(context->queue, params->gradient_buffer, CL_FALSE,
                             0, sizeof(float) * estimator->nr_of_dimensions,
                             gradient, estimator->nr_of_dimensions + 1, events,
                             &(result_events[0]));
  float error;
  err |= clEnqueueReadBuffer(context->queue, params->error_buffer, CL_FALSE,
                             0, sizeof(float), &error,
                             estimator->nr_of_dimensions + 1, events,
                             &(result_events[1]));
  // Wait for everything to finish.
  err |= clWaitForEvents(2, result_events);
  // All we have to do now is to normalize the error and the gradient.
  error /= params->nr_of_observations;
  fprintf(stderr, "\t>> grad:");
  for (i = 0; i<estimator->nr_of_dimensions; ++i) {
    gradient[i] /= params->nr_of_observations;
    // Apply a penalty for negative bandwidths.
    error += 0.05*exp(-30 * current_bandwidth[i]);
    gradient[i] -= 1.5*exp(-30 * current_bandwidth[i]);
    fprintf(stderr, " %f", gradient[i]);
  }
  fprintf(stderr, " - error: %f\n", error);
  // Ok, clean everything up.
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    clReleaseMemObject(sub_buffers[i]);
    clReleaseEvent(events[i]);
  }
  pfree(events);
  pfree(sub_buffers);
  clReleaseEvent(input_transfer_event);
  clReleaseEvent(partial_gradient_event);
  clReleaseEvent(result_events[0]);
  clReleaseEvent(result_events[1]);
  return error;
}

void ocl_runModelOptimization(ocl_estimator_t* estimator) {
  if (estimator == NULL) return;
  if (!kde_enable_bandwidth_optimization) return;
  fprintf(stderr, "Beginning model optimization for estimator on "
      "table %i\n", estimator->table);

  // First, we need to fetch the feedback records for this table and push them
  // to the device.
  cl_mem device_ranges, device_selectivites;
  unsigned int feedback_records = ocl_prepareFeedback(
      estimator, &device_ranges, &device_selectivites);
  if (feedback_records == 0) return;

  // Now prepare the optimizer.
  lbfgs_parameter_t optimizer_parameters;
  lbfgs_parameter_init(&optimizer_parameters);
  // We need to transfer the bandwidth to the host.
  float* host_bandwidth = lbfgs_malloc(estimator->nr_of_dimensions);
  ocl_context_t* context = ocl_getContext();
  clEnqueueReadBuffer(context->queue, estimator->bandwidth_buffer, CL_TRUE, 0,
                      sizeof(float) * estimator->nr_of_dimensions,
                      host_bandwidth, 0, NULL, NULL);
  // Package all required buffers.
  optimization_payload_t params;
  params.estimator = estimator;
  params.nr_of_observations = feedback_records;
  params.observed_ranges = device_ranges;
  params.observed_selectivities = device_selectivites;
  params.error_accumulator_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE, sizeof(float) * feedback_records,
      NULL, NULL);
  // Allocate a buffer to hold temporary gradient contributions. This buffer
  // will keep D contributions per observation. We store all contributions
  // consecutively (i.e. 111222333444). For optimal performance, we therefore
  // have to make sure that the consecutive regions (strides) have a size that
  // is aligned to the required machine alignment.
  params.stride_size = sizeof(float) * feedback_records;
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
      estimator->nr_of_dimensions * sizeof(float), NULL, NULL);
  params.error_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
  // Ok, we are prepared. Call bfgs to start the optimization.
  fprintf(stderr, "> Starting numerical optimization of bandwidth ... ");
  struct timeval start; gettimeofday(&start, NULL);
  evaluations = 0;
  int err = lbfgs(estimator->nr_of_dimensions, host_bandwidth, NULL,
        computeGradient, NULL, &params, &optimizer_parameters);
  struct timeval now; gettimeofday(&now, NULL);
  long seconds = now.tv_sec - start.tv_sec;
  long useconds = now.tv_usec - start.tv_usec;
  long mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
  // Ok, cool. Transfer the bandwidth back.
  fprintf(stderr, "done! Took %ld ms and %i gradient evaluations. Error %i.\n",
          mtime, evaluations, err);
  fprintf(stderr, "> Optimized bandwidth:");
  unsigned int i;
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
      fprintf(stderr, " %f", host_bandwidth[i]);
  }
  fprintf(stderr, "\n");
  clEnqueueWriteBuffer(context->queue, estimator->bandwidth_buffer, CL_TRUE, 0,
                       sizeof(float) * estimator->nr_of_dimensions,
                       host_bandwidth, 0, NULL, NULL);
  // Clean up.
  lbfgs_free(host_bandwidth);
  clReleaseMemObject(device_ranges);
  clReleaseMemObject(device_selectivites);
  clReleaseMemObject(params.error_buffer);
  clReleaseMemObject(params.gradient_buffer);
  clReleaseMemObject(params.error_accumulator_buffer);
  clReleaseMemObject(params.gradient_accumulator_buffer);
}
