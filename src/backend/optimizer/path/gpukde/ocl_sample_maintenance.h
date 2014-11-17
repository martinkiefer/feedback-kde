/*
 * ocl_sample_maintenance.h
 *
 *  Created on: 10.02.2014
 *      Author: mheimel
 */

#ifndef OCL_SAMPLE_MAINTENANCE_H_
#define OCL_SAMPLE_MAINTENANCE_H_

typedef enum ocl_estimator_quality_metric {
  IMPACT = 0,
  KARMA = 1
} ocl_estimator_quality_metric_t;

typedef struct ocl_sample_optimization {
  cl_mem sample_karma_buffer;     // Buffer to track the karma of the sample points.
  cl_mem sample_contribution_buffer; // Buffer to track the total probability contributions for the sample points.
  ocl_estimator_quality_metric_t last_optimized_sample_metric;
} ocl_sample_optimization_t;

void ocl_allocateSampleMaintenanceBuffers(ocl_estimator_t* estimator);
void ocl_releaseSampleMaintenanceBuffers(ocl_estimator_t* estimator);

void ocl_notifySampleMaintenanceOfSelectivity(
    ocl_estimator_t* estimator, double actual_selectivity);

#endif /* OCL_SAMPLE_MAINTENANCE_H_ */
