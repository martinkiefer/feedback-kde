/*
 * ocl_adaptive_bandwidth.h
 *
 *  Created on: 19.02.2014
 *      Author: mheimel
 */

#ifndef OCL_ADAPTIVE_BANDWIDTH_H_
#define OCL_ADAPTIVE_BANDWIDTH_H_

#include "ocl_estimator.h"

/**
 * Schedule the computation of required gradients for the online learning step.
 *
 * Returns an event to wait for the gradient computation to complete.
 */
void ocl_prepareOnlineLearningStep(ocl_estimator_t* estimator);

/*
 * Run a single online optimization step with adaptive learning rate.
 */
void ocl_runOnlineLearningStep(ocl_estimator_t* estimator,
                               double observed_selectivity);


#endif /* OCL_ADAPTIVE_BANDWIDTH_H_ */
