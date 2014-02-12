/*
 * ocl_model_maintenance.h
 *
 *  Created on: 11.02.2014
 *      Author: mheimel
 */

#ifndef OCL_MODEL_MAINTENANCE_H_
#define OCL_MODEL_MAINTENANCE_H_

#include "ocl_estimator.h"

// Checks the pg_kdefeedback table for potential feedback records to train
// this estimator. If feedback exists, the estimator starts a numerical
// optimization over these records to pick the optimal bandwidth.
void ocl_runModelOptimization(ocl_estimator_t* estimator);

#endif /* OCL_MODEL_MAINTENANCE_H_ */
