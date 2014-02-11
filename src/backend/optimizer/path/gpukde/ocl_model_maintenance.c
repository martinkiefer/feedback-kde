#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
/*
 * ocl_model_maintenance.c
 *
 *  Created on: 09.02.2014
 *      Author: mheimel
 */

#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#include "ocl_estimator.h"
#include "ocl_sample_maintenance.h"

// Global GUC variables
char* kde_estimation_quality_logfile_name;

// ############################################################
// # Define estimation error metrics.
// ############################################################

typedef struct error_metric {
	const char* name;
	float (*function)(float, float);
	float (*derivative_factor)(float, float);
} error_metric_t;


// Functions that implement the error metrics.
static float QuadraticError(float actual, float expected) {
  return (actual - expected) * (actual - expected);
}
static float QuadraticErrorDerivativeFactor(float actual, float expected) {
  return 2 * (actual - expected);
}
static float QErrror(float actual, float expected) {
  // Constants are required to avoid computing the log of 0.
  float tmp = log(0.001f + actual) - log(0.001f + expected);
  return tmp * tmp;
}
static float QErrorDerivativeFactor(float actual, float expected) {
  return 2 * (log(0.001f + actual) - log(0.001f + expected)) / (0.001f + actual);
}
static float AbsoluteError(float actual, float expected) {
  return fabs(actual - expected);
}
static float AbsoluteErrorDerivativeFactor(float actual, float expected) {
  if (actual > expected) {
    return 1;
  } else {
    return -1;
  }
}
static float RelativeError(float actual, float expected) {
  // Not entirely correct, but robust against zero estimates.
  return fabs(actual - expected) / (0.001f + expected);
}
static float RelativeErrorDerivativeFactor(float actual, float expected) {
  if (actual > expected) {
    return 1.0f / (0.001f + expected);
  } else {
    return -1.0f / (0.001f + expected);
  }
}

// Array of all available metrics.
static error_metric_t error_metrics[] = {
   {
      "Absolute",  &AbsoluteError, &AbsoluteErrorDerivativeFactor
   },
   {
      "Relative", &RelativeError, &RelativeErrorDerivativeFactor
   },
   {
      "Quadratic", &QuadraticError, &QuadraticErrorDerivativeFactor
   },
   {
      "Q", &QErrror, &QErrorDerivativeFactor
   }
};

typedef enum error_metrics {
  ABSOLUTE = 0,
  RELATIVE = 1,
  QUADRATIC = 2,
  Q = 3
} error_metrics_t;

error_metrics_t selected_metric = QUADRATIC;

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

// ############################################################
// # Code to compute the gradient for an observation.
// ############################################################

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
  cl_mem tmp_buffer = clCreateBuffer(context->context, CL_MEM_READ_WRITE, sizeof(float) * (processors + 1), NULL, NULL);
  err |= clSetKernelArg(init_buffer, 0, sizeof(cl_mem), &tmp_buffer);
  global_size = processors + 1;
  err |= clEnqueueNDRangeKernel(context->queue, init_buffer, 1, NULL, &global_size, NULL, 1, &external_event, &init_event);
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
    err |= clEnqueueNDRangeKernel(context->queue, fast_sum, 1, NULL, &global_size, &local_size, 1, &init_event, &event);
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

static void ocl_applySingleGradientStep(ocl_estimator_t* estimator, float selectivity) {
  cl_int err;
  ocl_context_t* context = ocl_getContext();

  struct timeval start; gettimeofday(&start, NULL);

  // Allocate a buffer to hold the temporary gradient contributions.
  size_t stride_size = sizeof(float) * estimator->rows_in_sample;
  if ((stride_size * 8) % context->required_mem_alignment) {
    // The stride size is misaligned, add some padding.
    stride_size *= 8;
    stride_size = (1 + stride_size / context->required_mem_alignment) * context->required_mem_alignment;
    stride_size /= 8;
  }
  unsigned int result_stride_elements = stride_size / sizeof(float);

  cl_mem gradient_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      stride_size * estimator->nr_of_dimensions, NULL, &err);
  if (gradient_buffer == NULL || err != CL_SUCCESS) {
    fprintf(stderr, "Could not allocate result buffer for gradient computation.\n");
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
  local_size = preferred_local_size_multiple * (local_size / preferred_local_size_multiple);
  // Ok, compute the global size to match this local size.
  size_t global_size = local_size * (estimator->rows_in_sample / local_size);
  if (global_size < estimator->rows_in_sample) global_size+=local_size;

  err |= clSetKernelArg(computeSingleGradient, 0, sizeof(cl_mem), &(estimator->sample_buffer));
  err |= clSetKernelArg(computeSingleGradient, 1, sizeof(unsigned int), &(estimator->rows_in_sample));
  err |= clSetKernelArg(computeSingleGradient, 2, sizeof(cl_mem), &(context->input_buffer));
  err |= clSetKernelArg(computeSingleGradient, 3, sizeof(cl_mem), &(estimator->bandwidth_buffer));
  err |= clSetKernelArg(computeSingleGradient, 4, available_local_memory, NULL);
  err |= clSetKernelArg(computeSingleGradient, 5, sizeof(cl_mem), &gradient_buffer);
  err |= clSetKernelArg(computeSingleGradient, 6, sizeof(unsigned int), &result_stride_elements);
  cl_event gradient_event;
  err |= clEnqueueNDRangeKernel(context->queue, computeSingleGradient, 1, NULL,
                                &global_size, &local_size, 0, NULL,
                                &gradient_event);

  // Ok cool, now compute the actual gradient.
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
  float scale_factor = 1.0f / (pow(2.0f, estimator->nr_of_dimensions) * estimator->rows_in_sample);
  scale_factor *= (*(error_metrics[selected_metric].derivative_factor))(estimator->last_selectivity, selectivity);
  scale_factor *= learning_rate;

  // And apply the gradient to the bandwidth.
  cl_kernel applyGradient = ocl_getKernel("applyGradient", 0);
  err |= clSetKernelArg(applyGradient, 0, sizeof(cl_mem), &(estimator->bandwidth_buffer));
  err |= clSetKernelArg(applyGradient, 1, sizeof(cl_mem), &gradient);
  err |= clSetKernelArg(applyGradient, 2, sizeof(float), &scale_factor);
  global_size = estimator->nr_of_dimensions;
  err |= clEnqueueNDRangeKernel(context->queue, applyGradient, 1, NULL,
                                &global_size, NULL, estimator->nr_of_dimensions, events, NULL);
  // Clean up.
  clFinish(context->queue);
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    clReleaseMemObject(sub_buffers[i]);
    clReleaseEvent(events[i]);
  }
  clReleaseMemObject(gradient_buffer);
  clReleaseEvent(gradient_event);
  pfree(events);
  pfree(sub_buffers);

  // Print timing:
  struct timeval now; gettimeofday(&now, NULL);
  long seconds = now.tv_sec - start.tv_sec;
  long useconds = now.tv_usec - start.tv_usec;
  long mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
  fprintf(stderr, "Adjusted bandwidth, took: %ld ms.\n", mtime);
}


// ############################################################
// # Main entry function.
// ############################################################
void ocl_notifyModelMaintenanceOfSelectivity(
    Oid relation, RQClause* bounds, float selectivity) {
  // Check if we have an estimator for this relation.
  ocl_estimator_t* estimator = ocl_getEstimator(relation);
  if (estimator == NULL) return;
  if (!estimator->open_estimation) return;  // No registered estimation.

  // Notify the sample maintenance of this observation so it can track the sample quality.
  //ocl_notifySampleMaintenanceOfSelectivity(estimator, selectivity);

  // Update the bandwidth.
  ocl_applySingleGradientStep(estimator, selectivity);

  // Write the error to the log file.
  ocl_reportErrorToLogFile(relation, estimator->last_selectivity, selectivity);

  // We are done.
  estimator->open_estimation = false;
}
