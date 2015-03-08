/*
 * ocl_sample_maintenance.h
 *
 *  Created on: 10.02.2014
 *      Author: mheimel
 */

#ifndef OCL_SAMPLE_MAINTENANCE_H_
#define OCL_SAMPLE_MAINTENANCE_H_

typedef struct ocl_sample_optimization {
  cl_mem sample_karma_buffer;     // Buffer to track the karma of the sample points.
  cl_mem sample_impact_buffer;	  // Buffer to track the impact of the sample points
  cl_mem sample_hitmap;		  //Working memory to identify qualifying sample points
  cl_mem deleted_point;		  //Working memory to store a deleted tuple for processing
  cl_mem min_val;		  //Working memory to store a minimum value
  cl_mem min_idx;		  //Working memory to store the index of a minimum value
} ocl_sample_optimization_t;

void ocl_allocateSampleMaintenanceBuffers(ocl_estimator_t* estimator);
void ocl_releaseSampleMaintenanceBuffers(ocl_estimator_t* estimator);

void ocl_notifySampleMaintenanceOfSelectivity(
    ocl_estimator_t* estimator, double actual_selectivity);

#endif /* OCL_SAMPLE_MAINTENANCE_H_ */
