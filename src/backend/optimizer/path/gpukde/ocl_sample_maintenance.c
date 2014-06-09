#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
/*
 * ocl_sample_maintenance.c
 *
 *  Created on: 07.02.2014
 *      Author: mheimel
 */

#include "ocl_estimator.h"
#include "ocl_utilities.h"
#include "optimizer/path/gpukde/ocl_estimator_api.h"

#include "catalog/pg_type.h"
#include "utils/rel.h"
#include "commands/vacuum.h"
#include "access/heapam.h"
#include "ocl_sample_maintenance.h"

#include <math.h>

#ifdef USE_OPENCL

extern ocl_kernel_type_t global_kernel_type;
// GUC configuration variable.
double kde_sample_maintenance_threshold;
double kde_sample_maintenance_exponential_decay;
int kde_sample_maintenance_period;
int kde_sample_maintenance_insert_option;
int kde_sample_maintenance_query_option;

//Convenience method for retrieving the index of the smallest element
static unsigned int getMinPenaltyIndex(ocl_context_t*  ctxt, ocl_estimator_t* estimator){
  cl_event event;
  unsigned int index = 0;
  
  //Allocate device memory for indizes and values
  cl_mem min_ix = clCreateBuffer(
          ctxt->context, CL_MEM_READ_WRITE,
          sizeof(unsigned int), NULL, NULL);
  cl_mem min_val = clCreateBuffer(
          ctxt->context, CL_MEM_READ_WRITE,
          sizeof(kde_float_t), NULL, NULL);
  
  event = minOfArray(estimator->sample_penalty_buffer, estimator->rows_in_sample,
                    min_val,min_ix, 0,
                    NULL);
    
  clEnqueueReadBuffer(ctxt->queue,min_ix, CL_TRUE, 0,
	                    sizeof(unsigned int), &index, 1, &event, NULL);  
  
  clReleaseEvent(event);
  return index;
}

//Helper method to get the minimum index below a certain threshold.
//Returns a NULL pointer if there is none.
static unsigned int *getMinPenaltyIndexBelowThreshold(ocl_context_t*  ctxt, ocl_estimator_t* estimator, kde_float_t threshold, cl_event wait_event){
  cl_event event;
  unsigned int *index = palloc(sizeof(unsigned int));
  kde_float_t val;
  
  //Allocate device memory for indizes and values
  cl_mem min_ix = clCreateBuffer(
          ctxt->context, CL_MEM_READ_WRITE,
          sizeof(unsigned int), NULL, NULL);
  cl_mem min_val = clCreateBuffer(
          ctxt->context, CL_MEM_READ_WRITE,
          sizeof(kde_float_t), NULL, NULL);
  
  event = minOfArray(estimator->sample_penalty_buffer, estimator->rows_in_sample,
                    min_val,min_ix, 0, wait_event);
  clEnqueueReadBuffer(ctxt->queue,min_ix, CL_TRUE, 0,
	               sizeof(kde_float_t), &val, 1, &event, NULL);
  if(val < threshold){
    clEnqueueReadBuffer(ctxt->queue,min_ix, CL_TRUE, 0,
	                    sizeof(unsigned int), index, 1, &event, NULL);
    clReleaseEvent(event);
    return index;
  }
  clReleaseEvent(event);
  return NULL; 
}

void ocl_notifySampleMaintenanceOfInsertion(Relation rel, HeapTuple new_tuple) {
  // Check whether we have a table for this relation.
  ocl_estimator_t* estimator = ocl_getEstimator(rel->rd_id);
  ocl_context_t * ctxt = ocl_getContext();
   
  if (estimator == NULL) return;
  estimator->rows_in_table++;
  if (! kde_sample_maintenance_insert_option) return;
  int insert_position = -1;
  
  // First, check whether we still have size in the sample.
  if (estimator->rows_in_sample < ocl_maxRowsInSample(estimator)) {
    insert_position = estimator->rows_in_sample++;
  } 
  else if(kde_sample_maintenance_insert_option == RESERVOIR){
    // The sample is full, use reservoir sampling.
    double rnd = (((double) random()) / ((double) MAX_RANDOM_VALUE));
    if (rnd > 1/(double) estimator->rows_in_table) return;
    insert_position = random() % estimator->rows_in_sample;
  }
  //TODO: This is not the best idea.
  //Think of something more flexible and clever.
  else if(kde_sample_maintenance_insert_option == RANDOM){
    // The sample is full, use random sampling
    double rnd = (((double) random()) / ((double) MAX_RANDOM_VALUE));
    if (rnd > 1/(double)estimator->rows_in_sample) return;
    insert_position = getMinPenaltyIndex(ctxt,estimator);
  }
  
  // Finally, extract the item and send it to the sample.
  kde_float_t* item = palloc(ocl_sizeOfSampleItem(estimator));
  ocl_extractSampleTuple(estimator, rel, new_tuple, item);
  ocl_pushEntryToSampleBufer(estimator, insert_position, item);
  pfree(item);
}

void ocl_notifySampleMaintenanceOfDeletion(Relation rel) {
  ocl_estimator_t* estimator = ocl_getEstimator(rel->rd_id);
  if (estimator == NULL) return;
  // For now, we just use this to update the table counts.
  estimator->rows_in_table--;
}

const double sample_match_learning_rate = 0.02f;

//For the moment, we will keep only track of the penalties here.
void ocl_notifySampleMaintenanceOfSelectivity(ocl_estimator_t* estimator,
                                              double actual_selectivity) {
  
  if (estimator == NULL) return;
  
  
  estimator->nr_of_estimations++;
  
  size_t global_size = estimator->rows_in_sample;
  cl_event penalty_event;

  
  kde_float_t normalization_factor;
  if(global_kernel_type == EPANECHNIKOV){
    normalization_factor = (kde_float_t) pow(0.75, estimator->nr_of_dimensions);
  }
  else {
    normalization_factor = (kde_float_t) pow(0.5, estimator->nr_of_dimensions);
  }

  cl_kernel kernel = ocl_getKernel("udate_sample_penalties_absolute",estimator->nr_of_dimensions);
  ocl_context_t * ctxt = ocl_getContext();
  kde_float_t samplesize_f = (kde_float_t) estimator->rows_in_sample;
  cl_int err = 0;
  err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &(ctxt->result_buffer));
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &(estimator->sample_penalty_buffer));
  err |= clSetKernelArg(kernel, 2, sizeof(kde_float_t), &(samplesize_f));
  err |= clSetKernelArg(kernel, 3, sizeof(kde_float_t), &(normalization_factor));
  err |= clSetKernelArg(kernel, 4, sizeof(kde_float_t), &(estimator->last_selectivity));
  err |= clSetKernelArg(kernel, 5, sizeof(kde_float_t), &(actual_selectivity));
  err |= clSetKernelArg(kernel, 6, sizeof(kde_float_t), &(kde_sample_maintenance_exponential_decay));
  
  err |= clEnqueueNDRangeKernel(ctxt->queue, kernel, 1, NULL, &global_size,
	                         NULL, 0, NULL, &penalty_event);
  
  clReleaseEvent(penalty_event);  
  
  //TODO: Insert sample maintenance query code here.
}



#endif
