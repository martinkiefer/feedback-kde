#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
/*
 * ocl_sample_maintenance.c
 *
 *  Created on: 07.02.2014
 *      Author: mheimel
 */

#include "ocl_estimator.h"
#include "ocl_utilities.h"

#include "catalog/pg_type.h"
#include "utils/rel.h"

#ifdef USE_OPENCL

// GUC configuration variable.
bool kde_propagate_inserts;

void ocl_notifySampleMaintenanceOfInsertion(Relation rel, HeapTuple new_tuple) {
  // Check whether we have a table for this relation.
  ocl_estimator_t* estimator = ocl_getEstimator(rel->rd_id);
  if (estimator == NULL) return;
  estimator->rows_in_table++;
  if (!kde_propagate_inserts) return;
  int insert_position = -1;
  // First, check whether we still have size in the sample.
  if (estimator->rows_in_sample < ocl_maxRowsInSample(estimator)) {
    insert_position = estimator->rows_in_sample++;
  } else {
    // The sample is full, use reservoir sampling.
    double rnd = (((double) random()) / ((double) MAX_RANDOM_VALUE));
    if (rnd > 1/(double)estimator->rows_in_table) return;
    insert_position = random() % estimator->rows_in_sample;
  }
  // Finally, extract the item and send it to the sample.
  float* item = palloc(ocl_sizeOfSampleItem(estimator));
  ocl_extractSampleTuple(estimator, rel->rd_id, new_tuple, item);
  ocl_pushEntryToSampleBufer(estimator, insert_position, item);
  pfree(item);
}

void ocl_notifySampleMaintenanceOfDeletion(Relation rel) {
  ocl_estimator_t* estimator = ocl_getEstimator(rel->rd_id);
  if (estimator == NULL) return;
  // For now, we just use this to update the table counts.
  estimator->rows_in_table--;
}

const float sample_match_learning_rate = 0.02f;

void ocl_notifySampleMaintenanceOfSelectivity(ocl_estimator_t* estimator,
                                              float actual_selectivity) {
  ocl_context_t* context = ocl_getContext();
  cl_kernel kernel = ocl_getKernel("update_sample_quality_metrics", 0);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &(context->result_buffer));
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &(estimator->sample_quality_slopes_buffer));
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &(estimator->sample_quality_intercepts_buffer));
  clSetKernelArg(kernel, 3, sizeof(float), &actual_selectivity);
  clSetKernelArg(kernel, 4, sizeof(float), &sample_match_learning_rate);
  size_t global_size = estimator->rows_in_sample;
  cl_int err = clEnqueueNDRangeKernel(context->queue, kernel, 1, NULL,
                         &global_size, NULL, 0, NULL, NULL);
  clFinish(context->queue);

  // Cleanup.
  clReleaseKernel(kernel);
}

#endif
