#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
/*
 * ocl_adaptive_bandwidth.c
 *
 *  Created on: 19.02.2014
 *      Author: mheimel
 */

#include "ocl_adaptive_bandwidth.h"

#include <math.h>

#include "ocl_estimator.h"
#include "ocl_error_metrics.h"
#include "ocl_utilities.h"

cl_kernel init_zero = NULL;
cl_kernel init_one = NULL;
cl_kernel computePartialGradient = NULL;  
cl_kernel finalizeKernel = NULL;
cl_kernel accumulate = NULL;
cl_kernel updateModel = NULL;
cl_kernel initModel = NULL;

// GUC configuration variables.
bool kde_enable_adaptive_bandwidth;
int kde_adaptive_bandwidth_minibatch_size;
int kde_online_optimization_algorithm;

void ocl_allocateBandwidthOptimizatztionBuffers(ocl_estimator_t* estimator) {
  ocl_bandwidth_optimization_t* descriptor = calloc(
      1, sizeof(ocl_bandwidth_optimization_t));
  descriptor->learning_boost_rate = 10;
  // We are done.
  estimator->bandwidth_optimization = descriptor;
}

static void ocl_initializeVsgdBuffersForOnlineLearning(
    ocl_estimator_t* estimator) {
  cl_int err = CL_SUCCESS;
  ocl_context_t* context = ocl_getContext();
  ocl_bandwidth_optimization_t* descriptor = estimator->bandwidth_optimization;

  // Prepare the initialization kernels.
  if (init_zero == NULL) init_zero = ocl_getKernel("init_zero", 0);
  if (init_one == NULL) init_one = ocl_getKernel("init_one", 0);
  size_t global_size = estimator->nr_of_dimensions;

  // Initialize the accumulator buffers and fill them with zero.
  descriptor->gradient_accumulator = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  err = clSetKernelArg(
      init_zero, 0, sizeof(cl_mem), &(descriptor->gradient_accumulator));
  Assert(err == CL_SUCCESS);
  err = clEnqueueNDRangeKernel(
      context->queue, init_zero, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);

  descriptor->squared_gradient_accumulator = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  err = clSetKernelArg(
      init_zero, 0, sizeof(cl_mem),
      &(descriptor->squared_gradient_accumulator));
  Assert(err == CL_SUCCESS);
  err = clEnqueueNDRangeKernel(
      context->queue, init_zero, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  
  descriptor->hessian_accumulator = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  err = clSetKernelArg(
      init_zero, 0, sizeof(cl_mem), &(descriptor->hessian_accumulator));
  Assert(err == CL_SUCCESS);
  err = clEnqueueNDRangeKernel(
      context->queue, init_zero, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);

  descriptor->squared_hessian_accumulator = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  err = clSetKernelArg(
      init_zero, 0, sizeof(cl_mem), &(descriptor->squared_hessian_accumulator));
  Assert(err == CL_SUCCESS);
  err = clEnqueueNDRangeKernel(
      context->queue, init_zero, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  
  // Initialize the running average buffers with zero.
  descriptor->running_gradient_average = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  // We initialize the gradient average buffer with a small positive value,
  // since this gradient is used by the online estimator to determine the
  // finite difference step to estimate the Hessian.
  err = clSetKernelArg(
      init_one, 0, sizeof(cl_mem), &(descriptor->running_gradient_average));
  Assert(err == CL_SUCCESS);
  err = clEnqueueNDRangeKernel(
      context->queue, init_one, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);

  descriptor->running_squared_gradient_average = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  err = clSetKernelArg(
      init_zero, 0, sizeof(cl_mem),
      &(descriptor->running_squared_gradient_average));
  Assert(err == CL_SUCCESS);
  err = clEnqueueNDRangeKernel(
      context->queue, init_zero, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);

  descriptor->running_hessian_average = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  err = clSetKernelArg(
      init_zero, 0, sizeof(cl_mem), &(descriptor->running_hessian_average));
  Assert(err == CL_SUCCESS);
  err = clEnqueueNDRangeKernel(
      context->queue, init_zero, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);

  descriptor->running_squared_hessian_average = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  err = clSetKernelArg(
      init_zero, 0, sizeof(cl_mem),
      &(descriptor->running_squared_hessian_average));
  Assert(err == CL_SUCCESS);
  err = clEnqueueNDRangeKernel(
      context->queue, init_zero, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  
  // Initialize the time constant buffer to two.
  descriptor->current_time_constant = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  err = clSetKernelArg(
      init_one, 0, sizeof(cl_mem), &(descriptor->current_time_constant));
  Assert(err == CL_SUCCESS);
  err = clEnqueueNDRangeKernel(
      context->queue, init_one, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);

  // Allocate the buffers to compute temporary gradients.
  descriptor->temp_gradient_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  descriptor->temp_shifted_gradient_buffer= clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  descriptor->temp_shifted_result_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE, sizeof(kde_float_t), NULL, &err);
  Assert(err == CL_SUCCESS);

  // And finish.
  clFinish(context->queue);
}

/**
 * Schedule the computation of the gradient at the current bandwidth. We
 * also compute a shifted gradient (by a small delta) to estimate the Hessian
 * curvature.
 */
static void ocl_prepareVsgdOnlineLearningStep(ocl_estimator_t* estimator) {
  unsigned int i;
  cl_int err = CL_SUCCESS;
  ocl_context_t* context = ocl_getContext();
  cl_mem null_buffer = NULL;
  ocl_bandwidth_optimization_t* descriptor = estimator->bandwidth_optimization;

  // Ensure that all required buffers are set up.
  if (descriptor->gradient_accumulator == NULL) {
    ocl_initializeVsgdBuffersForOnlineLearning(estimator);
  }

  cl_event* summation_events = palloc(
      sizeof(cl_event) * (2 * estimator->nr_of_dimensions + 1));

  // Compute the required stride size for the partial gradient buffers.
  size_t stride_size = sizeof(kde_float_t) * estimator->rows_in_sample;
  if ((stride_size * 8) % context->required_mem_alignment) {
    // The stride size is not aligned, add some padding.
    stride_size *= 8;
    stride_size = (1 + stride_size / context->required_mem_alignment)
                          * context->required_mem_alignment;
    stride_size /= 8;
  }
  unsigned int result_stride_elements = stride_size / sizeof(kde_float_t);

  // Figure out the optimal local size for the partial gradient kernel.
  if (computePartialGradient == NULL) {
    computePartialGradient = ocl_getKernel(
        "computePartialGradient", estimator->nr_of_dimensions);
  }
  // We start with the maximum supporter local size.
  size_t local_size;
  err = clGetKernelWorkGroupInfo(
      computePartialGradient, context->device, CL_KERNEL_WORK_GROUP_SIZE,
      sizeof(size_t), &local_size, NULL);
  Assert(err == CL_SUCCESS);
  // Then we cap this to the local memory requirements.
  size_t available_local_memory;
  err = clGetKernelWorkGroupInfo(
      computePartialGradient, context->device, CL_KERNEL_LOCAL_MEM_SIZE,
      sizeof(size_t), &available_local_memory, NULL);
  Assert(err == CL_SUCCESS);
  available_local_memory = context->local_mem_size - available_local_memory;
  local_size = Min(
      local_size,
      available_local_memory / (sizeof(kde_float_t) * estimator->nr_of_dimensions));
  // And finally ensure that the local size is a multiple of the preferred size.
  size_t preferred_local_size_multiple;
  err = clGetKernelWorkGroupInfo(
      computePartialGradient, context->device,
      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
      sizeof(size_t), &preferred_local_size_multiple, NULL);
  Assert(err == CL_SUCCESS);
  local_size = preferred_local_size_multiple
      * (local_size / preferred_local_size_multiple);

  // Ensure that the global size is big enough to accomodate all sample items.
  size_t global_size = local_size * (estimator->rows_in_sample / local_size);
  if (global_size < estimator->rows_in_sample) global_size += local_size;

    // Schedule the computation of the partial gradient for the current bandwidth.
  cl_event partial_gradient_event = NULL;
  cl_mem partial_gradient_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      stride_size * estimator->nr_of_dimensions, NULL, &err);
  
  // Set the common parameters for the partial gradient computations.
  err |= clSetKernelArg(
      computePartialGradient, 0, sizeof(cl_mem), &(estimator->sample_buffer));
  err |= clSetKernelArg(
      computePartialGradient, 1, sizeof(unsigned int),
      &(estimator->rows_in_sample));
  err |= clSetKernelArg(
      computePartialGradient, 2, sizeof(cl_mem), &(estimator->input_buffer));
  err |= clSetKernelArg(
      computePartialGradient, 3, sizeof(cl_mem),
      &(estimator->bandwidth_buffer));
  err |= clSetKernelArg(
      computePartialGradient, 5, available_local_memory, NULL);
  err |= clSetKernelArg(
      computePartialGradient, 7, sizeof(unsigned int), &result_stride_elements);
  err |= clSetKernelArg(
      computePartialGradient, 4, sizeof(cl_mem), &null_buffer);
  err |= clSetKernelArg(
      computePartialGradient, 6, sizeof(cl_mem), &partial_gradient_buffer);
  err |= clSetKernelArg(
      computePartialGradient, 8, sizeof(cl_mem), &null_buffer);
  Assert(err == CL_SUCCESS);
  
  err = clEnqueueNDRangeKernel(
      context->queue, computePartialGradient, 1, NULL,
      &global_size, &local_size, 0, NULL, &partial_gradient_event);
  Assert(err == CL_SUCCESS);
  
  // Now schedule the summation of the partial gradient computations.
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    cl_buffer_region region;
    region.size = stride_size;
    region.origin = i * stride_size;
    cl_mem gradient_sub_buffer = clCreateSubBuffer(
        partial_gradient_buffer, CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    summation_events[i] = sumOfArray(
        gradient_sub_buffer, estimator->rows_in_sample,
        descriptor->temp_gradient_buffer, i, partial_gradient_event);
    err = clReleaseMemObject(gradient_sub_buffer);
    Assert(err == CL_SUCCESS);
  }
  err = clReleaseEvent(partial_gradient_event);
  Assert(err == CL_SUCCESS);
  
  // Schedule the computation of the partial gradient for the shifted bandwidth.
  cl_event partial_shifted_gradient_event = NULL;
  cl_mem partial_shifted_gradient_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      stride_size * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  cl_mem partial_shifted_result_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->rows_in_sample, NULL, &err);
  Assert(err == CL_SUCCESS);
  err |= clSetKernelArg(computePartialGradient, 4, sizeof(cl_mem),
      &(descriptor->running_gradient_average));
  err |= clSetKernelArg(computePartialGradient, 6, sizeof(cl_mem),
      &partial_shifted_gradient_buffer);
  err |= clSetKernelArg(computePartialGradient, 8, sizeof(cl_mem),
      &partial_shifted_result_buffer);
  Assert(err == CL_SUCCESS);
  
  err = clEnqueueNDRangeKernel(context->queue, computePartialGradient, 1,
      NULL, &global_size, &local_size, 0, NULL,
      &partial_shifted_gradient_event);

  // Schedule the summation of the partial shifted gradient contributions.
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    cl_buffer_region region;
    region.size = stride_size;
    region.origin = i * stride_size;
    cl_mem shifted_gradient_sub_buffer = clCreateSubBuffer(
        partial_shifted_gradient_buffer, CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    Assert(err == CL_SUCCESS);
    summation_events[estimator->nr_of_dimensions + i] = sumOfArray(
        shifted_gradient_sub_buffer, estimator->rows_in_sample,
        descriptor->temp_shifted_gradient_buffer, i,
        partial_shifted_gradient_event);
    err = clReleaseMemObject(shifted_gradient_sub_buffer);
    Assert(err == CL_SUCCESS);
  }
  // Also schedule the kernel that computes the sum of local result contributions
  // for the shifted gradient.
  summation_events[2 * estimator->nr_of_dimensions] = sumOfArray(
      partial_shifted_result_buffer, estimator->rows_in_sample,
      descriptor->temp_shifted_result_buffer, 0, partial_shifted_gradient_event);
  err = clReleaseEvent(partial_shifted_gradient_event);
  Assert(err == CL_SUCCESS);

  ocl_printBuffer(
      " > Gradient at bandwidth:", descriptor->temp_gradient_buffer,
      estimator->nr_of_dimensions, 1);
  ocl_printBuffer(
      " > Shifted gradient:", descriptor->temp_shifted_gradient_buffer,
      estimator->nr_of_dimensions, 1);

  // Finally, schedule a single task that serves as an indicator that all computations
  // have finished and that normalizes the shifted gradient result.
  kde_float_t normalization_factor =
      pow(0.5, estimator->nr_of_dimensions) / estimator->rows_in_sample;
  if (finalizeKernel == NULL) finalizeKernel = ocl_getKernel("finalizeEstimate", 0);
  err |= clSetKernelArg(
      finalizeKernel, 0, sizeof(cl_mem),
      &(descriptor->temp_shifted_result_buffer));
  err |= clSetKernelArg(
      finalizeKernel, 1, sizeof(kde_float_t), &normalization_factor);
  Assert(err == CL_SUCCESS);
  err = clEnqueueTask(
      context->queue, finalizeKernel,
      2*estimator->nr_of_dimensions + 1,
      summation_events, &(descriptor->optimization_event));
  Assert(err == CL_SUCCESS);
  
  // Clean up.
  for (i=0; i<(2 * estimator->nr_of_dimensions + 1); ++i) {
    err = clReleaseEvent(summation_events[i]);
    Assert(err == CL_SUCCESS);
  }
  pfree(summation_events);
  if (partial_gradient_buffer)
    err |= clReleaseMemObject(partial_gradient_buffer);
  if (partial_shifted_gradient_buffer)
    err |= clReleaseMemObject(partial_shifted_gradient_buffer);
  if (partial_shifted_result_buffer)
    err |= clReleaseMemObject(partial_shifted_result_buffer);
  Assert(err == CL_SUCCESS);
}

static void ocl_runVsgdOnlineLearningStep(
    ocl_estimator_t* estimator, double selectivity) {
  if (!kde_enable_adaptive_bandwidth) return;
  if (ocl_isDebug()) fprintf(stderr, ">>> Running online learning step.\n");
  ocl_bandwidth_optimization_t* descriptor = estimator->bandwidth_optimization;

  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;

  // Fetch the estimate for the shifted bandwidth.
  kde_float_t shifted_estimate;
  err = clEnqueueReadBuffer(
      context->queue, descriptor->temp_shifted_result_buffer, CL_TRUE,
      0, sizeof(kde_float_t), &shifted_estimate, 1,
      &(descriptor->optimization_event), NULL);
  Assert(err == CL_SUCCESS);
  err = clReleaseEvent(descriptor->optimization_event);
  Assert(err == CL_SUCCESS);
  descriptor->optimization_event = NULL;

  // Compute the scaling factor for the gradient.
  kde_float_t gradient_factor =
      M_SQRT2 / (sqrt(M_PI) * pow(2.0, estimator->nr_of_dimensions));
  gradient_factor *= (*(ocl_getSelectedErrorMetric()->gradient_factor))(
      estimator->last_selectivity, selectivity, estimator->rows_in_table);
  gradient_factor *= estimator->rows_in_sample;
  // Compute the scaling factor for the adjusted gradient.
  kde_float_t shifted_gradient_factor =
      M_SQRT2 / (sqrt(M_PI) * pow(2.0, estimator->nr_of_dimensions));
  shifted_gradient_factor *= (*(ocl_getSelectedErrorMetric()->gradient_factor))(
      shifted_estimate, selectivity, estimator->rows_in_table);
  shifted_gradient_factor *= estimator->rows_in_sample;
  size_t global_size = estimator->nr_of_dimensions;

  // Now accumulate the gradient within the current mini-batch.
  if (accumulate == NULL) {
    accumulate = ocl_getKernel("accumulateVsgdOnlineBuffers", 0);
  }
  err |= clSetKernelArg(
      accumulate, 0, sizeof(cl_mem), &(descriptor->temp_gradient_buffer));
  err |= clSetKernelArg(
      accumulate, 1, sizeof(cl_mem),
      &(descriptor->temp_shifted_gradient_buffer));
  err |= clSetKernelArg(
      accumulate, 2, sizeof(kde_float_t), &gradient_factor);
  err |= clSetKernelArg(
      accumulate, 3, sizeof(kde_float_t), &shifted_gradient_factor);
  err |= clSetKernelArg(
      accumulate, 4, sizeof(cl_mem), &(estimator->bandwidth_buffer));
  err |= clSetKernelArg(
      accumulate, 5, sizeof(cl_mem), &(descriptor->running_gradient_average));
  err |= clSetKernelArg(
      accumulate, 6, sizeof(cl_mem), &(descriptor->gradient_accumulator));
  err |= clSetKernelArg(
      accumulate, 7, sizeof(cl_mem),
      &(descriptor->squared_gradient_accumulator));
  err |= clSetKernelArg(
      accumulate, 8, sizeof(cl_mem), &(descriptor->hessian_accumulator));
  err |= clSetKernelArg(
      accumulate, 9, sizeof(cl_mem),
      &(descriptor->squared_hessian_accumulator));
  Assert(err == CL_SUCCESS);
  cl_event accumulator_event;
  err = clEnqueueNDRangeKernel(
      context->queue, accumulate, 1, NULL, &global_size, NULL, 0, NULL,
      &accumulator_event);
  Assert(err == CL_SUCCESS);
  
  // Debug print the accumulated buffers.
  ocl_printBuffer(
      "\tAccumulated gradient:", descriptor->gradient_accumulator,
      estimator->nr_of_dimensions, 1);
  ocl_printBuffer(
      "\tAccumulated gradient^2:", descriptor->squared_gradient_accumulator,
      estimator->nr_of_dimensions, 1);
  ocl_printBuffer(
      "\tAccumulated hessian:", descriptor->hessian_accumulator,
      estimator->nr_of_dimensions, 1);
  ocl_printBuffer(
      "\tAccumulated hessian^2:", descriptor->squared_hessian_accumulator,
      estimator->nr_of_dimensions, 1);

  descriptor->nr_of_accumulated_gradients++;
  // Check if we have a full mini-batch.
  if (descriptor->nr_of_accumulated_gradients >= kde_adaptive_bandwidth_minibatch_size) {
    if (ocl_isDebug()) fprintf(stderr, "\t >> Full minibatch <<\n");
    if (descriptor->online_learning_initialized) {
      // If we are initialized, compute the next bandwidth.
      if (updateModel == NULL) {
        updateModel = ocl_getKernel("updateVsgdOnlineEstimate", 0);
      }
      err |= clSetKernelArg(
          updateModel, 0, sizeof(cl_mem), &(descriptor->gradient_accumulator));
      err |= clSetKernelArg(
          updateModel, 1, sizeof(cl_mem),
          &(descriptor->squared_gradient_accumulator));
      err |= clSetKernelArg(
          updateModel, 2, sizeof(cl_mem), &(descriptor->hessian_accumulator));
      err |= clSetKernelArg(
          updateModel, 3, sizeof(cl_mem),
          &(descriptor->squared_hessian_accumulator));
      err |= clSetKernelArg(
          updateModel, 4, sizeof(cl_mem),
          &(descriptor->running_gradient_average));
      err |= clSetKernelArg(
          updateModel, 5, sizeof(cl_mem),
          &(descriptor->running_squared_gradient_average));
      err |= clSetKernelArg(
          updateModel, 6, sizeof(cl_mem),
          &(descriptor->running_hessian_average));
      err |= clSetKernelArg(
          updateModel, 7, sizeof(cl_mem),
          &(descriptor->running_squared_hessian_average));
      err |= clSetKernelArg(
          updateModel, 8, sizeof(cl_mem), &(descriptor->current_time_constant));
      err |= clSetKernelArg(
          updateModel, 9, sizeof(cl_mem), &(estimator->bandwidth_buffer));
      err |= clSetKernelArg(
          updateModel, 10, sizeof(unsigned int),
          &kde_adaptive_bandwidth_minibatch_size);
      err |= clSetKernelArg(
          updateModel, 11, sizeof(kde_float_t),
          &(descriptor->learning_boost_rate));
      Assert(err == CL_SUCCESS);
      err = clEnqueueNDRangeKernel(
          context->queue, updateModel, 1, NULL, &global_size, NULL, 1,
          &accumulator_event, &(descriptor->optimization_event));
      Assert(err == CL_SUCCESS);
    } else {
      // In order to initialize the algorithm, we simply use the accumulated averages.
      if (initModel == NULL) {
        initModel = ocl_getKernel("initializeVsgdOnlineEstimate", 0);
      }
      err |= clSetKernelArg(
          initModel, 0, sizeof(cl_mem), &(descriptor->gradient_accumulator));
      err |= clSetKernelArg(
          initModel, 1, sizeof(cl_mem),
          &(descriptor->squared_gradient_accumulator ));
      err |= clSetKernelArg(
          initModel, 2, sizeof(cl_mem), &(descriptor->hessian_accumulator));
      err |= clSetKernelArg(
          initModel, 3, sizeof(cl_mem),
          &(descriptor->squared_hessian_accumulator));
      err |= clSetKernelArg(
          initModel, 4, sizeof(cl_mem),
          &(descriptor->running_gradient_average));
      err |= clSetKernelArg(
          initModel, 5, sizeof(cl_mem),
          &(descriptor->running_squared_gradient_average));
      err |= clSetKernelArg(
          initModel, 6, sizeof(cl_mem), &(descriptor->running_hessian_average));
      err |= clSetKernelArg(
          initModel, 7, sizeof(cl_mem),
          &(descriptor->running_squared_hessian_average));
      err |= clSetKernelArg(
          initModel, 8, sizeof(cl_mem), &(descriptor->current_time_constant));
      err |= clSetKernelArg(
          initModel, 9, sizeof(unsigned int),
          &kde_adaptive_bandwidth_minibatch_size);
      Assert(err == CL_SUCCESS);
      err = clEnqueueNDRangeKernel(
          context->queue, initModel, 1, NULL, &global_size, NULL, 1,
          &accumulator_event, &(descriptor->optimization_event));
      Assert(err == CL_SUCCESS);
      descriptor->online_learning_initialized = true;
    }
    descriptor->nr_of_accumulated_gradients = 0;

    // Debug print the accumulated buffers.
    ocl_printBuffer(
        "\tTime-averaged gradient:", descriptor->running_gradient_average,
        estimator->nr_of_dimensions, 1);
    ocl_printBuffer(
        "\tTime-averaged gradient^2:",
        descriptor->running_squared_gradient_average,
        estimator->nr_of_dimensions, 1);
    ocl_printBuffer(
        "\tTime-averaged hessian:", descriptor->running_hessian_average,
        estimator->nr_of_dimensions, 1);
    ocl_printBuffer(
        "\tTime-averaged hessian^2:",
        descriptor->running_squared_hessian_average,
        estimator->nr_of_dimensions, 1);
    ocl_printBuffer("\tUpdated bandwidth:", estimator->bandwidth_buffer,
        estimator->nr_of_dimensions, 1);
  }

  // Clean up.
  err = clReleaseEvent(accumulator_event);
  Assert(err == CL_SUCCESS);
}

// Helper function to initialize rmsprop learning for a given estimator.
static void ocl_initializeRMSProp(
    ocl_estimator_t* estimator) {
  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  ocl_rmsPropOptimization_t* descriptor = calloc(
      1, sizeof(ocl_rmsPropOptimization_t));
  cl_mem nullBuffer = NULL;
  unsigned int i = 0;

  /*
   * Initialize the partial gradient computation.
   */
  // Figure out the required stride size for storing the partial gradient
  // contributions and allocate the corresponding buffer.
  size_t stride_size = sizeof(kde_float_t) * estimator->rows_in_sample;
  if ((stride_size * 8) % context->required_mem_alignment) {
    // The stride size is not aligned, add some padding.
    stride_size *= 8;
    stride_size = (1 + stride_size / context->required_mem_alignment) * context->required_mem_alignment;
    stride_size /= 8;
  }
  unsigned int result_stride_elements = stride_size / sizeof(kde_float_t);
  descriptor->partial_gradient_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      stride_size * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  // Create the kernel, and decide on its optimal local and global sizes.
  descriptor->compute_partial_gradient =
        ocl_getKernel("computePartialGradient", estimator->nr_of_dimensions);
  err = clGetKernelWorkGroupInfo(
      descriptor->compute_partial_gradient, context->device,
      CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
      &(descriptor->partial_gradient_localsize), NULL);
  Assert(err == CL_SUCCESS);
  
  // The optimal local size is decided by a) the available local memory
  // and b) the preferred workgroup size multiple.
  size_t available_local_memory;
  err = clGetKernelWorkGroupInfo(
      descriptor->compute_partial_gradient, context->device,
      CL_KERNEL_LOCAL_MEM_SIZE, sizeof(size_t), &available_local_memory, NULL);
  Assert(err == CL_SUCCESS);
  available_local_memory = context->local_mem_size - available_local_memory;
  descriptor->partial_gradient_localsize = Min(
      descriptor->partial_gradient_localsize,
      available_local_memory / (sizeof(kde_float_t) * estimator->nr_of_dimensions));
  size_t preferred_local_size_multiple;
  err = clGetKernelWorkGroupInfo(
      descriptor->compute_partial_gradient, context->device,
      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t),
      &preferred_local_size_multiple, NULL);
  Assert(err == CL_SUCCESS);
  descriptor->partial_gradient_localsize = preferred_local_size_multiple
      * (descriptor->partial_gradient_localsize / preferred_local_size_multiple);
  // The global size is chosen to ensure we cover all sample items.
  descriptor->partial_gradient_globalsize =
      descriptor->partial_gradient_localsize * (estimator->rows_in_sample / descriptor->partial_gradient_localsize);
  if (descriptor->partial_gradient_globalsize < estimator->rows_in_sample) {
    descriptor->partial_gradient_globalsize += descriptor->partial_gradient_localsize;
  }
  // Now set the kernel options.
  err |= clSetKernelArg(
      descriptor->compute_partial_gradient, 0, sizeof(cl_mem),
      &(estimator->sample_buffer));
  err |= clSetKernelArg(
      descriptor->compute_partial_gradient, 1, sizeof(unsigned int),
      &(estimator->rows_in_sample));
  err |= clSetKernelArg(
      descriptor->compute_partial_gradient, 2, sizeof(cl_mem),
      &(estimator->input_buffer));
  err |= clSetKernelArg(
      descriptor->compute_partial_gradient, 3, sizeof(cl_mem),
      &(estimator->bandwidth_buffer));
  err |= clSetKernelArg(
      descriptor->compute_partial_gradient, 4, sizeof(cl_mem), &nullBuffer);
  err |= clSetKernelArg(
      descriptor->compute_partial_gradient, 5, descriptor->partial_gradient_localsize
      *estimator->nr_of_dimensions * sizeof(kde_float_t), NULL);
  err |= clSetKernelArg(
      descriptor->compute_partial_gradient, 6, sizeof(cl_mem),
      &(descriptor->partial_gradient_buffer));
  err |= clSetKernelArg(
      descriptor->compute_partial_gradient, 7, sizeof(unsigned int),
      &result_stride_elements);
  err |= clSetKernelArg(
      descriptor->compute_partial_gradient, 8, sizeof(cl_mem), &nullBuffer);
  Assert(err == CL_SUCCESS);
  
  /*
   * Initialize the partial gradient summation.
   */
  descriptor->gradient_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      estimator->nr_of_dimensions * sizeof(kde_float_t), NULL, &err);
  Assert(err == CL_SUCCESS);
  descriptor->gradient_summation_buffers = calloc(
      1, sizeof(cl_mem) * estimator->nr_of_dimensions);
  descriptor->gradient_summation_descriptors = calloc(
      1, sizeof(ocl_aggregation_descriptor_t) * estimator->nr_of_dimensions);
  descriptor->gradient_summation_events = calloc(
      1, sizeof(cl_mem) * estimator->nr_of_dimensions);
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    cl_buffer_region region;
    region.size = stride_size;
    region.origin = i * stride_size;
    descriptor->gradient_summation_buffers[i] = clCreateSubBuffer(
        descriptor->partial_gradient_buffer, CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION, &region, NULL);
    descriptor->gradient_summation_descriptors[i] = prepareSumDescriptor(
        descriptor->gradient_summation_buffers[i], estimator->rows_in_sample,
        descriptor->gradient_buffer, i);
  }

  /*
   * Initialize the gradient accumulation.
   */
  // Allocate and zero-initialize the gradient accumulator.
  descriptor->gradient_accumulator_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  cl_kernel init_zero = ocl_getKernel("init_zero", 0);
  size_t global_size = estimator->nr_of_dimensions;
  err = clSetKernelArg(
      init_zero, 0, sizeof(cl_mem), &(descriptor->gradient_accumulator_buffer));
  Assert(err == CL_SUCCESS);
  err = clEnqueueNDRangeKernel(
      context->queue, init_zero, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  err = clReleaseKernel(init_zero);
  Assert(err == CL_SUCCESS);
  // Prepare the accumulation kernel.
  descriptor->gradient_accumulator = ocl_getKernel(
      "accumulateRmspropOnlineBuffers", 0);
  err |= clSetKernelArg(
      descriptor->gradient_accumulator, 0, sizeof(cl_mem),
      &(descriptor->gradient_buffer));
  err |= clSetKernelArg(
      descriptor->gradient_accumulator, 2, sizeof(cl_mem),
      &(estimator->bandwidth_buffer));
  err |= clSetKernelArg(
      descriptor->gradient_accumulator, 3, sizeof(cl_mem),
      &(descriptor->gradient_accumulator_buffer));

  /*
   * Initialize the learning rate adjustment buffers (will be initialized
   * when the first mini-batch is accumulated.)
   */
  descriptor->last_gradient_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  descriptor->running_squared_gradient_average_buffer = clCreateBuffer(
        context->context, CL_MEM_READ_WRITE,
        sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);
  descriptor->learning_rate_buffer = clCreateBuffer(
        context->context, CL_MEM_READ_WRITE,
        sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);

  /*
   * Initialize the update kernel.
   */
  descriptor->model_update = ocl_getKernel("updateRmspropOnlineEstimate", 0);
  err |= clSetKernelArg(
      descriptor->model_update, 0, sizeof(cl_mem),
      &(descriptor->gradient_accumulator_buffer));
  err |= clSetKernelArg(
      descriptor->model_update, 1, sizeof(cl_mem),
      &(descriptor->last_gradient_buffer));
  err |= clSetKernelArg(
      descriptor->model_update, 2, sizeof(cl_mem),
      &(descriptor->running_squared_gradient_average_buffer));
  err |= clSetKernelArg(
      descriptor->model_update, 3, sizeof(cl_mem),
      &(descriptor->learning_rate_buffer));
  err |= clSetKernelArg(
      descriptor->model_update, 4, sizeof(cl_mem),
      &(estimator->bandwidth_buffer));
  err |= clSetKernelArg(
      descriptor->model_update, 5, sizeof(unsigned int),
      &kde_adaptive_bandwidth_minibatch_size);
  Assert(err == CL_SUCCESS);
  
  // We are done :)
  estimator->bandwidth_optimization->rmsprop_descriptor = descriptor;
  err = clFinish(context->queue);
  Assert(err == CL_SUCCESS);
}

// Helper function to release an initialized rmsprop descriptor.
static void ocl_releaseRMSProp(ocl_estimator_t* estimator) {
  unsigned int i;
  cl_int err = CL_SUCCESS;
  // Sanity checks.
  if (! estimator->bandwidth_optimization) return;
  if (! estimator->bandwidth_optimization->rmsprop_descriptor) return;
  // Ok, clean up the descriptor.
  ocl_rmsPropOptimization_t* descriptor =
      estimator->bandwidth_optimization->rmsprop_descriptor;
  err = clReleaseKernel(descriptor->compute_partial_gradient);
  Assert(err == CL_SUCCESS);
  err = clReleaseMemObject(descriptor->gradient_buffer);
  Assert(err == CL_SUCCESS);
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    err = clReleaseMemObject(descriptor->gradient_summation_buffers[i]);
    Assert(err == CL_SUCCESS);
    releaseAggregationDescriptor(descriptor->gradient_summation_descriptors[i]);
  }
  free(descriptor->gradient_summation_buffers);
  free(descriptor->gradient_summation_descriptors);
  free(descriptor->gradient_summation_events);
  err |= clReleaseMemObject(descriptor->partial_gradient_buffer);
  err |= clReleaseMemObject(descriptor->gradient_accumulator_buffer);
  err |= clReleaseKernel(descriptor->gradient_accumulator);
  err |= clReleaseMemObject(descriptor->last_gradient_buffer);
  err |= clReleaseMemObject(descriptor->running_squared_gradient_average_buffer);
  err |= clReleaseMemObject(descriptor->learning_rate_buffer);
  err |= clReleaseKernel(descriptor->model_update);
  Assert(err == CL_SUCCESS);
  
  // We are done.
  free(estimator->bandwidth_optimization->rmsprop_descriptor);
  estimator->bandwidth_optimization->rmsprop_descriptor = NULL;
}
// Helper function to schedule the computation of the partial gradient.
static void ocl_prepareRmspropOnlineLearningStep(ocl_estimator_t* estimator) {
  unsigned int i;
  cl_int err = CL_SUCCESS;
  ocl_context_t* context = ocl_getContext();
  if (! estimator->bandwidth_optimization->rmsprop_descriptor) {
    ocl_initializeRMSProp(estimator);
  }
  ocl_rmsPropOptimization_t* descriptor =
      estimator->bandwidth_optimization->rmsprop_descriptor;

  // Schedule the computation of the partial gradient contributions for the
  // current bandwidth.
  cl_event partial_gradient_event = NULL;
  err = clEnqueueNDRangeKernel(
      context->queue, descriptor->compute_partial_gradient, 1, NULL,
      &(descriptor->partial_gradient_globalsize),
      &(descriptor->partial_gradient_localsize), 0, NULL,
      &partial_gradient_event);
  Assert(err == CL_SUCCESS);
  // Now schedule the summation of the partial gradient computations.
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    descriptor->gradient_summation_events[i] = predefinedSumOfArray(
        descriptor->gradient_summation_descriptors[i], partial_gradient_event);
  }
  err = clReleaseEvent(partial_gradient_event);
  Assert(err == CL_SUCCESS);
}

static void ocl_runRmspropOnlineLearningStep(
    ocl_estimator_t* estimator, double selectivity) {
  //Keeps as much of the learning code the same.
  if (!kde_enable_adaptive_bandwidth) return;

  if (ocl_isDebug()) fprintf(stderr, ">>> Running online learning step.\n");
  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  ocl_rmsPropOptimization_t* descriptor =
      estimator->bandwidth_optimization->rmsprop_descriptor;
  ocl_printBuffer(
      "\tGradient at bandwidth:",
      descriptor->gradient_buffer, estimator->nr_of_dimensions, 1);

  // Compute the scaling factor for the gradient.
  kde_float_t gradient_factor =
      M_SQRT2 / (sqrt(M_PI) * pow(2.0, estimator->nr_of_dimensions));
  gradient_factor *= (*(ocl_getSelectedErrorMetric()->gradient_factor))(
      estimator->last_selectivity, selectivity, estimator->rows_in_table);
  gradient_factor *= estimator->rows_in_sample;
  // Now call the accumulator with the computed scaling factor.
  err = clSetKernelArg(
      descriptor->gradient_accumulator, 1, sizeof(kde_float_t),
      &gradient_factor);
  Assert(err == CL_SUCCESS);
  size_t global_size = estimator->nr_of_dimensions;
  cl_event accumulator_event;
  err = clEnqueueNDRangeKernel(
      context->queue, descriptor->gradient_accumulator, 1, NULL, &global_size,
      NULL, estimator->nr_of_dimensions, descriptor->gradient_summation_events,
      &accumulator_event);
  Assert(err == CL_SUCCESS);
  // Debug print the accumulated buffers.
  ocl_printBuffer(
      "\tAccumulated gradient:", descriptor->gradient_accumulator_buffer,
      estimator->nr_of_dimensions, 1);

  // Ok, we have a new observation, check if we have a full mini-batch:
  if (++(descriptor->accumulated_gradients) % kde_adaptive_bandwidth_minibatch_size == 0) {
    // Full mini-batch:
    if (ocl_isDebug()) fprintf(stderr, "\t >> Full minibatch <<\n");
    if (! descriptor->optimization_initialized) {
      // In order to initialize the algorithm, we simply use the accumulated averages.
      cl_kernel initModel = ocl_getKernel("initializeRmspropOnlineEstimate", 0);
      err |= clSetKernelArg(
          initModel, 0, sizeof(cl_mem), &(descriptor->gradient_accumulator_buffer));
      err |= clSetKernelArg(
          initModel, 1, sizeof(cl_mem), &(descriptor->last_gradient_buffer));
      err |= clSetKernelArg(
          initModel, 2, sizeof(cl_mem), &(descriptor->learning_rate_buffer));
      err |= clSetKernelArg(
          initModel, 3, sizeof(cl_mem),
          &(descriptor->running_squared_gradient_average_buffer));
      err |= clSetKernelArg(
          initModel, 4, sizeof(unsigned int),
          &kde_adaptive_bandwidth_minibatch_size);
      Assert(err == CL_SUCCESS);
      err = clEnqueueNDRangeKernel(
          context->queue, initModel, 1, NULL, &global_size, NULL, 1,
          &accumulator_event, NULL);
      Assert(err == CL_SUCCESS);
      err = clReleaseKernel(initModel);
      Assert(err == CL_SUCCESS);
      descriptor->optimization_initialized = true;
      err = clFinish(context->queue);
      Assert(err == CL_SUCCESS);
    }
    // Schedule the mini-batch update.
    err = clEnqueueNDRangeKernel(
        context->queue, descriptor->model_update, 1, NULL, &global_size,
        NULL, 1, &accumulator_event,
        &(estimator->bandwidth_optimization->optimization_event));
    Assert(err == CL_SUCCESS);
    // Debug print the accumulated buffers.
    ocl_printBuffer(
        "\tUpdated bandwidth:", estimator->bandwidth_buffer,
        estimator->nr_of_dimensions, 1);
  }
  err = clReleaseEvent(accumulator_event);
  Assert(err == CL_SUCCESS);
}

void ocl_runOnlineLearningStep(
    ocl_estimator_t* estimator, double selectivity) {
  if (!kde_enable_adaptive_bandwidth) return;

  if(kde_online_optimization_algorithm == VSGD_FD) {
    ocl_runVsgdOnlineLearningStep(estimator, selectivity);
  } else if (kde_online_optimization_algorithm == RMSPROP) {
    ocl_runRmspropOnlineLearningStep(estimator, selectivity);
  } else {
    fprintf(
        stderr, "I do not know this optimization algorithm: %i\n",
        kde_online_optimization_algorithm);
  }
}

void ocl_prepareOnlineLearningStep(ocl_estimator_t* estimator) {
  if (!kde_enable_adaptive_bandwidth) return;
  if(kde_online_optimization_algorithm == VSGD_FD) {
    ocl_prepareVsgdOnlineLearningStep(estimator);
  } else if(kde_online_optimization_algorithm == RMSPROP) {
    ocl_prepareRmspropOnlineLearningStep(estimator);
  } else {
    fprintf(
        stderr, "I do not know this optimization algorithm: %i\n",
        kde_online_optimization_algorithm);
  }
}

void ocl_releaseBandwidthOptimizatztionBuffers(ocl_estimator_t* estimator) {
  cl_int err = CL_SUCCESS;
  if (! estimator->bandwidth_optimization) return;
  ocl_bandwidth_optimization_t* descriptor = estimator->bandwidth_optimization;
  if (estimator->bandwidth_optimization->rmsprop_descriptor) {
    ocl_releaseRMSProp(estimator);
  }

  // Release the buffers.
  if (descriptor->gradient_accumulator) {
    err |= clReleaseMemObject(descriptor->gradient_accumulator);
  }
  if (descriptor->squared_gradient_accumulator) {
    err |= clReleaseMemObject(descriptor->squared_gradient_accumulator);
  }
  if (descriptor->hessian_accumulator) {
    err |= clReleaseMemObject(descriptor->hessian_accumulator);
  }
  if (descriptor->squared_hessian_accumulator) {
    err |= clReleaseMemObject(descriptor->squared_hessian_accumulator);
  }
  if (descriptor->running_gradient_average) {
    err |= clReleaseMemObject(descriptor->running_gradient_average);
  }
  if (descriptor->running_squared_gradient_average) {
    err |= clReleaseMemObject(descriptor->running_squared_gradient_average);
  }
  if (descriptor->running_hessian_average) {
    err |= clReleaseMemObject(descriptor->running_hessian_average);
  }
  if (descriptor->running_squared_hessian_average) {
    err |= clReleaseMemObject(descriptor->running_squared_hessian_average);
  }
  if (descriptor->current_time_constant) {
    err |= clReleaseMemObject(descriptor->current_time_constant);
  }
  if (descriptor->temp_gradient_buffer) {
    err |= clReleaseMemObject(descriptor->temp_gradient_buffer);
  }
  if (descriptor->temp_shifted_gradient_buffer) {
    err |= clReleaseMemObject(descriptor->temp_shifted_gradient_buffer);
  }
  if (descriptor->temp_shifted_result_buffer) {
    err |= clReleaseMemObject(descriptor->temp_shifted_result_buffer);
  }
  if (descriptor->last_gradient) {
    err |= clReleaseMemObject(descriptor->last_gradient);
  }
  if (descriptor->learning_rate) {
    err |= clReleaseMemObject(descriptor->learning_rate);
  }
  Assert(err == CL_SUCCESS);
  // We are done :)
  free(descriptor);
}

