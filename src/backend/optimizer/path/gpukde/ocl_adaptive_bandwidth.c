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

// GUC configuration variables.
bool kde_enable_adaptive_bandwidth;
int kde_adaptive_bandwidth_minibatch_size;

static void ocl_initializeBuffersForOnlineLearning(ocl_estimator_t* estimator) {
  cl_int err = CL_SUCCESS;
  ocl_context_t* context = ocl_getContext();

  // Prepare the initialization kernels.
  cl_kernel init_zero = ocl_getKernel("init_zero", 0);
  cl_kernel init_one = ocl_getKernel("init_one", 0);
  cl_kernel init_small = ocl_getKernel("init", 0);
  kde_float_t small_val = 1e-4;
  err |= clSetKernelArg(init_small, 1, sizeof(kde_float_t), &small_val);
  size_t global_size = estimator->nr_of_dimensions;

  // Initialize the accumulator buffers and fill them with zero.

  estimator->gradient_accumulator = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);
  err |= clSetKernelArg(init_zero, 0, sizeof(cl_mem),
                        &(estimator->gradient_accumulator));
  err |= clEnqueueNDRangeKernel(context->queue, init_zero, 1, NULL,
                                &global_size, NULL, 0, NULL, NULL);

  estimator->squared_gradient_accumulator = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);
  err |= clSetKernelArg(init_zero, 0, sizeof(cl_mem),
                        &(estimator->squared_gradient_accumulator));
  err |= clEnqueueNDRangeKernel(context->queue, init_zero, 1, NULL,
                                &global_size, NULL, 0, NULL, NULL);

  estimator->hessian_accumulator = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);
  err |= clSetKernelArg(init_zero, 0, sizeof(cl_mem),
                        &(estimator->hessian_accumulator));
  err |= clEnqueueNDRangeKernel(context->queue, init_zero, 1, NULL,
                                &global_size, NULL, 0, NULL, NULL);

  estimator->squared_hessian_accumulator = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);
  err |= clSetKernelArg(init_zero, 0, sizeof(cl_mem),
                        &(estimator->squared_hessian_accumulator));
  err |= clEnqueueNDRangeKernel(context->queue, init_zero, 1, NULL,
                                &global_size, NULL, 0, NULL, NULL);

  // Initialize the running average buffers with zero.

  estimator->running_gradient_average = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);
    // We initialize the gradient average buffer with a small positive value,
    // since this gradient is used by the online estimator to determine the
    // finite difference step to estimate the Hessian.
  err |= clSetKernelArg(init_small, 0, sizeof(cl_mem),
                        &(estimator->running_gradient_average));
  err |= clEnqueueNDRangeKernel(context->queue, init_small, 1, NULL,
                                &global_size, NULL, 0, NULL, NULL);

  estimator->running_squared_gradient_average = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);
  err |= clSetKernelArg(init_zero, 0, sizeof(cl_mem),
                        &(estimator->running_squared_gradient_average));
  err |= clEnqueueNDRangeKernel(context->queue, init_zero, 1, NULL,
                                &global_size, NULL, 0, NULL, NULL);

  estimator->running_hessian_average = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);
  err |= clSetKernelArg(init_zero, 0, sizeof(cl_mem),
                        &(estimator->running_hessian_average));
  err |= clEnqueueNDRangeKernel(context->queue, init_zero, 1, NULL,
                                &global_size, NULL, 0, NULL, NULL);

  estimator->running_squared_hessian_average = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);
  err |= clSetKernelArg(init_zero, 0, sizeof(cl_mem),
                        &(estimator->running_squared_hessian_average));
  err |= clEnqueueNDRangeKernel(context->queue, init_zero, 1, NULL,
                                &global_size, NULL, 0, NULL, NULL);

  // Initialize the time constant buffer with one.
  estimator->current_time_constant = clCreateBuffer(
       context->context, CL_MEM_READ_WRITE,
       sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);
   err |= clSetKernelArg(init_one, 0, sizeof(cl_mem),
                         &(estimator->current_time_constant));
   err |= clEnqueueNDRangeKernel(context->queue, init_one, 1, NULL,
                                 &global_size, NULL, 0, NULL, NULL);

   // Allocate the buffers to compute temporary gradients.
   estimator->temp_gradient_buffer = clCreateBuffer(
       context->context, CL_MEM_READ_WRITE,
       sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);
   estimator->temp_hessian_buffer = clCreateBuffer(
       context->context, CL_MEM_READ_WRITE,
       sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, NULL);

  // Clean up and wait for everything to finish..
  clReleaseKernel(init_one);
  clReleaseKernel(init_zero);
  clReleaseKernel(init_small);
  // And finish.
  clFinish(context->queue);
}

/**
 * Schedule the computation of the gradient at the current bandwidth. We
 * also compute a shifted gradient (by a small delta) to estimate the Hessian
 * curvature.
 */
void ocl_prepareOnlineLearningStep(ocl_estimator_t* estimator) {
  if (!kde_enable_adaptive_bandwidth) return;

  unsigned int i;
  cl_int err = CL_SUCCESS;
  ocl_context_t* context = ocl_getContext();
  cl_mem null_buffer = NULL;

  // Ensure that all required buffers are set up.
  if (estimator->gradient_accumulator == NULL) {
    ocl_initializeBuffersForOnlineLearning(estimator);
  }

  cl_event* summation_events = palloc(
      2 * sizeof(cl_event) * estimator->nr_of_dimensions);

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
  cl_kernel computePartialGradient = ocl_getKernel(
      "computePartialGradient", estimator->nr_of_dimensions);
    // We start with the maximum supporter local size.
  size_t local_size;
  clGetKernelWorkGroupInfo(computePartialGradient, context->device,
                           CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                           &local_size, NULL);
    // Then we cap this to the local memory requirements.
  size_t available_local_memory;
  clGetKernelWorkGroupInfo(computePartialGradient, context->device,
                           CL_KERNEL_LOCAL_MEM_SIZE, sizeof(size_t),
                           &available_local_memory, NULL);
  available_local_memory = context->local_mem_size - available_local_memory;
  local_size = Min(
      local_size,
      available_local_memory / (sizeof(kde_float_t) * estimator->nr_of_dimensions));
    // And finally ensure that the local size is a multiple of the preferred size.
  size_t preferred_local_size_multiple;
  clGetKernelWorkGroupInfo(computePartialGradient, context->device,
                           CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                           sizeof(size_t), &preferred_local_size_multiple, NULL);
  local_size = preferred_local_size_multiple
      * (local_size / preferred_local_size_multiple);

  // Now ensure that the global size is big enoug to accomodate all sample items.
  size_t global_size = local_size * (estimator->rows_in_sample / local_size);
  if (global_size < estimator->rows_in_sample) global_size += local_size;

  // Set the common parameters for the partial gradient computations.
  err |= clSetKernelArg(computePartialGradient, 0, sizeof(cl_mem),
                        &(estimator->sample_buffer));
  err |= clSetKernelArg(computePartialGradient, 1, sizeof(unsigned int),
                        &(estimator->rows_in_sample));
  err |= clSetKernelArg(computePartialGradient, 2, sizeof(cl_mem),
                        &(context->input_buffer));
  err |= clSetKernelArg(computePartialGradient, 3, sizeof(cl_mem),
                        &(estimator->bandwidth_buffer));
  err |= clSetKernelArg(computePartialGradient, 5, available_local_memory, NULL);
  err |= clSetKernelArg(computePartialGradient, 7, sizeof(unsigned int),
                        &result_stride_elements);

  // Schedule the computation of the partial gradient for the current bandwidth.
  cl_event partial_gradient_event = NULL;
  cl_mem partial_gradient_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      stride_size * estimator->nr_of_dimensions, NULL, &err);
  err |= clSetKernelArg(computePartialGradient, 4, sizeof(cl_mem),
                        &null_buffer);
  err |= clSetKernelArg(computePartialGradient, 6, sizeof(cl_mem),
                        &partial_gradient_buffer);
  err |= clEnqueueNDRangeKernel(context->queue, computePartialGradient, 1,
                                NULL, &global_size, &local_size, 0, NULL,
                                &partial_gradient_event);

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
        estimator->temp_gradient_buffer, i, partial_gradient_event);
    clReleaseMemObject(gradient_sub_buffer);
  }
  clReleaseEvent(partial_gradient_event);

  // Schedule the computation of the partial gradient for the shifted bandwidth.
  cl_event partial_shifted_gradient_event = NULL;
  cl_mem partial_shifted_gradient_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      stride_size * estimator->nr_of_dimensions, NULL, &err);
  err |= clSetKernelArg(computePartialGradient, 4, sizeof(cl_mem),
                        &(estimator->running_gradient_average));
  err |= clSetKernelArg(computePartialGradient, 6, sizeof(cl_mem),
                        &partial_shifted_gradient_buffer);
  err |= clEnqueueNDRangeKernel(context->queue, computePartialGradient, 1,
                                NULL, &global_size, &local_size, 0, NULL,
                                &partial_shifted_gradient_event);

  // Now schedule the summation of the partial shifted gradient contributions.
  for (i=0; i<estimator->nr_of_dimensions; ++i) {
    cl_buffer_region region;
    region.size = stride_size;
    region.origin = i * stride_size;
    cl_mem shifted_gradient_sub_buffer = clCreateSubBuffer(
        partial_shifted_gradient_buffer, CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    summation_events[estimator->nr_of_dimensions + i] = sumOfArray(
        shifted_gradient_sub_buffer, estimator->rows_in_sample,
        estimator->temp_hessian_buffer, i, partial_shifted_gradient_event);
    clReleaseMemObject(shifted_gradient_sub_buffer);
  }
  clReleaseEvent(partial_shifted_gradient_event);

  // Finally, schedule the kernel that computes the finite-difference
  // estimation of the Hessian.
  cl_kernel computeHessian = ocl_getKernel("computeFDHessian", 0);
  global_size = estimator->nr_of_dimensions;
  err |= clSetKernelArg(computeHessian, 0, sizeof(cl_mem),
                        &(estimator->temp_gradient_buffer));
  err |= clSetKernelArg(computeHessian, 1, sizeof(cl_mem),
                        &(estimator->running_gradient_average));
  err |= clSetKernelArg(computeHessian, 2, sizeof(cl_mem),
                        &(estimator->temp_hessian_buffer));
  err |= clEnqueueNDRangeKernel(
      context->queue, computeHessian, 1, NULL, &global_size, NULL,
      estimator->nr_of_dimensions, summation_events,
      &(estimator->temp_gradient_event));

  // Clean up.
  clReleaseKernel(computeHessian);
  clReleaseKernel(computePartialGradient);
  for (i=0; i<(estimator->online_learning_initialized ? 2 : 1) * estimator->nr_of_dimensions; ++i) {
    clReleaseEvent(summation_events[i]);
  }
  pfree(summation_events);
  if (partial_gradient_buffer)
      clReleaseMemObject(partial_gradient_buffer);
  if (partial_shifted_gradient_buffer)
    clReleaseMemObject(partial_shifted_gradient_buffer);
}

void ocl_runOnlineLearningStep(ocl_estimator_t* estimator,
                               double selectivity) {
  if (!kde_enable_adaptive_bandwidth) return;

  if (ocl_isDebug()) fprintf(stderr, ">>> Running online learning step.\n");

  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;

  // Compute the factor to adjust the gradient.
  kde_float_t gradient_factor = 1.0f
      / (pow(2.0f, estimator->nr_of_dimensions) * estimator->rows_in_sample);
  gradient_factor *= (*(ocl_getSelectedErrorMetric()->gradient_factor))(
      estimator->last_selectivity, selectivity, estimator->rows_in_table);

  size_t global_size = estimator->nr_of_dimensions;

  // Now accumulate the gradient within the current mini-batch.
  cl_kernel accumulate = ocl_getKernel("accumulateOnlineBuffers", 0);
  err |= clSetKernelArg(accumulate, 0, sizeof(cl_mem),
                        &(estimator->temp_gradient_buffer));
  err |= clSetKernelArg(accumulate, 1, sizeof(cl_mem),
                        &(estimator->temp_hessian_buffer));
  err |= clSetKernelArg(accumulate, 2, sizeof(kde_float_t), &gradient_factor);
  err |= clSetKernelArg(accumulate, 3, sizeof(cl_mem),
                        &(estimator->gradient_accumulator));
  err |= clSetKernelArg(accumulate, 4, sizeof(cl_mem),
                        &(estimator->squared_gradient_accumulator));
  err |= clSetKernelArg(accumulate, 5, sizeof(cl_mem),
                        &(estimator->hessian_accumulator));
  err |= clSetKernelArg(accumulate, 6, sizeof(cl_mem),
                        &(estimator->squared_hessian_accumulator));
  cl_event accumulator_event;
  err |= clEnqueueNDRangeKernel(
      context->queue, accumulate, 1, NULL, &global_size, NULL, 1,
      &(estimator->temp_gradient_event), &accumulator_event);

  // Debug print the accumulated buffers.
  ocl_printBuffer("\tTemp gradient:", estimator->temp_gradient_buffer,
                  estimator->nr_of_dimensions, 1);
  ocl_printBuffer("\tTemp hessian:", estimator->temp_hessian_buffer,
                  estimator->nr_of_dimensions, 1);
  ocl_printBuffer("\tAccumulated gradient:", estimator->gradient_accumulator,
                  estimator->nr_of_dimensions, 1);
  ocl_printBuffer("\tAccumulated gradient^2:", estimator->squared_gradient_accumulator,
                  estimator->nr_of_dimensions, 1);
  ocl_printBuffer("\tAccumulated hessian:", estimator->hessian_accumulator,
                  estimator->nr_of_dimensions, 1);
  ocl_printBuffer("\tAccumulated hessian^2:", estimator->squared_hessian_accumulator,
                  estimator->nr_of_dimensions, 1);

  estimator->nr_of_accumulated_gradients++;
  // Check if we have a full mini-batch.
  if (estimator->nr_of_accumulated_gradients >= kde_adaptive_bandwidth_minibatch_size) {
    if (ocl_isDebug()) fprintf(stderr, "\t >> Full minibatch <<\n");

    if (estimator->online_learning_initialized) {
      // If we are initialized, compute the next bandwidth.
      cl_kernel updateModel = ocl_getKernel("updateOnlineEstimate", 0);
      err |= clSetKernelArg(updateModel, 0, sizeof(cl_mem),
                            &(estimator->gradient_accumulator));
      err |= clSetKernelArg(updateModel, 1, sizeof(cl_mem),
                            &(estimator->squared_gradient_accumulator));
      err |= clSetKernelArg(updateModel, 2, sizeof(cl_mem),
                            &(estimator->hessian_accumulator));
      err |= clSetKernelArg(updateModel, 3, sizeof(cl_mem),
                            &(estimator->squared_hessian_accumulator));
      err |= clSetKernelArg(updateModel, 4, sizeof(cl_mem),
                            &(estimator->running_gradient_average));
      err |= clSetKernelArg(updateModel, 5, sizeof(cl_mem),
                            &(estimator->running_squared_gradient_average));
      err |= clSetKernelArg(updateModel, 6, sizeof(cl_mem),
                            &(estimator->running_hessian_average));
      err |= clSetKernelArg(updateModel, 7, sizeof(cl_mem),
                            &(estimator->running_squared_hessian_average));
      err |= clSetKernelArg(updateModel, 8, sizeof(cl_mem),
                            &(estimator->current_time_constant));
      err |= clSetKernelArg(updateModel, 9, sizeof(cl_mem),
                            &(estimator->bandwidth_buffer));
      err |= clSetKernelArg(updateModel, 10, sizeof(unsigned int),
                            &kde_adaptive_bandwidth_minibatch_size);
      err |= clEnqueueNDRangeKernel(
          context->queue, updateModel, 1, NULL, &global_size, NULL, 1,
          &accumulator_event, NULL);
      clReleaseKernel(updateModel);
    } else {
      // In order to initialize the algorithm, we simply use the accumulated averages.
      cl_kernel initModel = ocl_getKernel("initializeOnlineEstimate", 0);
      err |= clSetKernelArg(initModel, 0, sizeof(cl_mem),
                            &(estimator->gradient_accumulator));
      err |= clSetKernelArg(initModel, 1, sizeof(cl_mem),
                            &(estimator->squared_gradient_accumulator ));
      err |= clSetKernelArg(initModel, 2, sizeof(cl_mem),
                            &(estimator->hessian_accumulator));
      err |= clSetKernelArg(initModel, 3, sizeof(cl_mem),
                            &(estimator->squared_hessian_accumulator));
      err |= clSetKernelArg(initModel, 4, sizeof(cl_mem),
                            &(estimator->running_gradient_average));
      err |= clSetKernelArg(initModel, 5, sizeof(cl_mem),
                            &(estimator->running_squared_gradient_average));
      err |= clSetKernelArg(initModel, 6, sizeof(cl_mem),
                            &(estimator->running_hessian_average));
      err |= clSetKernelArg(initModel, 7, sizeof(cl_mem),
                            &(estimator->running_squared_hessian_average));
      err |= clSetKernelArg(initModel, 8, sizeof(unsigned int),
                            &kde_adaptive_bandwidth_minibatch_size);
      err |= clEnqueueNDRangeKernel(
          context->queue, initModel, 1, NULL, &global_size, NULL, 1,
          &accumulator_event, NULL);
      clReleaseKernel(initModel);
      estimator->online_learning_initialized = true;

    }
    estimator->nr_of_accumulated_gradients = 0;

    // Debug print the accumulated buffers.
    ocl_printBuffer("\tTime-averaged gradient:", estimator->running_gradient_average,
                    estimator->nr_of_dimensions, 1);
    ocl_printBuffer("\tTime-averaged gradient^2:", estimator->running_squared_gradient_average,
                    estimator->nr_of_dimensions, 1);
    ocl_printBuffer("\tTime-averaged hessian:", estimator->running_hessian_average,
                    estimator->nr_of_dimensions, 1);
    ocl_printBuffer("\tTime-averaged hessian^2:", estimator->running_squared_hessian_average,
                    estimator->nr_of_dimensions, 1);
    ocl_printBuffer("\tUpdated bandwidth:", estimator->bandwidth_buffer,
                    estimator->nr_of_dimensions, 1);
  }

  // Wait.
  clFinish(context->queue);

  // Clean up.
  clReleaseKernel(accumulate);
  clReleaseEvent(estimator->temp_gradient_event);
  estimator->temp_gradient_event = NULL;
  clReleaseEvent(accumulator_event);

}
