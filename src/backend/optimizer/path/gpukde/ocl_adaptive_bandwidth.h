/*
 * ocl_adaptive_bandwidth.h
 *
 *  Created on: 19.02.2014
 *      Author: mheimel
 */

#ifndef OCL_ADAPTIVE_BANDWIDTH_H_
#define OCL_ADAPTIVE_BANDWIDTH_H_

#include "ocl_estimator.h"

// Required fields for rmsprop.
typedef struct ocl_rmsPropOptimization {
  // Kernel to compute the partial gradient computations.
  cl_kernel compute_partial_gradient;
  size_t partial_gradient_localsize;
  size_t partial_gradient_globalsize;
  // Buffer to store the partial gradient computations.
  cl_mem partial_gradient_buffer;
  // Buffer to store the gradient for the current observation.
  cl_mem gradient_buffer;
  // Summation descriptors and sub-buffers for the gradient summation.
  cl_mem* gradient_summation_buffers;
  ocl_aggregation_descriptor_t** gradient_summation_descriptors;
  cl_event* gradient_summation_events;
  // Buffer and kernel to accumulate partial gradients.
  cl_mem gradient_accumulator_buffer;
  cl_kernel gradient_accumulator;
  // Buffers for the learning rate adjustment.
  cl_mem last_gradient_buffer;
  cl_mem running_squared_gradient_average_buffer;
  cl_mem learning_rate_buffer;
  // Kernel to run the model update.
  cl_kernel model_update;
  // Trackers for the current state of the optimization.
  unsigned int accumulated_gradients;
  char optimization_initialized;
} ocl_rmsPropOptimization_t;

typedef struct ocl_bandwidth_optimization {
  // Event that is used by the estimator to wait for an ongoing model
  // optimization.
  cl_event optimization_event;
  // Fields for RMSprop learning.
  ocl_rmsPropOptimization_t* rmsprop_descriptor;
  // Fields for svgd-fd learning.
  cl_mem gradient_accumulator;
  cl_mem squared_gradient_accumulator;
  cl_mem hessian_accumulator;
  cl_mem squared_hessian_accumulator;
  unsigned int nr_of_accumulated_gradients;
  /* Fields for computing the adaptive learning rate */
  bool online_learning_initialized;
  cl_mem last_gradient;
  cl_mem learning_rate;
  cl_mem running_gradient_average;
  cl_mem running_squared_gradient_average;
  cl_mem running_hessian_average;
  cl_mem running_squared_hessian_average;
  cl_mem current_time_constant;
  /* Fields for pre-computing the gradient for online learning */
  cl_mem temp_gradient_buffer;
  cl_mem temp_shifted_gradient_buffer;
  cl_mem temp_shifted_result_buffer;
  double learning_boost_rate;
  // Fields for describing aggregation operations.
  cl_mem partial_gradient_buffer;
} ocl_bandwidth_optimization_t;

void ocl_allocateBandwidthOptimizatztionBuffers(ocl_estimator_t* estimator);
void ocl_releaseBandwidthOptimizatztionBuffers(ocl_estimator_t* estimator);

/**
 * Schedule the computation of required gradients for the online learning step.
 *
 * Returns an event to wait for the gradient computation to complete.
 */
void ocl_prepareOnlineLearningStep(ocl_estimator_t* estimator);

/*
 * Run a single online optimization step with adaptive learning rate.
 */
void ocl_runOnlineLearningStep(
    ocl_estimator_t* estimator, double observed_selectivity);


#endif /* OCL_ADAPTIVE_BANDWIDTH_H_ */
