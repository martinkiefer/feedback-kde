/*
 * ocl_adaptive_bandwidth.h
 *
 *  Created on: 19.02.2014
 *      Author: mheimel
 */

#ifndef OCL_ADAPTIVE_BANDWIDTH_H_
#define OCL_ADAPTIVE_BANDWIDTH_H_

#include "ocl_estimator.h"

typedef struct ocl_bandwidth_optimization {
  /* Fields for tracking mini-batch updates to the bandwidth. */
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
  cl_event online_learning_event;
  double learning_boost_rate;
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
