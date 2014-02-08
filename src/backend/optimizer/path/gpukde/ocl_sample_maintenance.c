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

/*
 * Global GUC Config variables.
 */
bool ocl_propagate_inserts_to_sample;

/**
 *
 */
void ocl_notifyEstimatorOfInsertion(Relation rel, HeapTuple new_tuple) {
  if (!ocl_useKDE()) return;
  // Check whether we have a table for this relation.
  ocl_estimator_t* estimator = ocl_getEstimator(rel);
  if (estimator == NULL) return;
  estimator->rows_in_table++;
  if (!ocl_propagate_inserts_to_sample) return;
  int insert_position = -1;
  // First, check whether we still have size in the sample.
  int max_rows_in_sample = estimator->sample_buffer_size /
      ocl_sizeOfSampleItem(estimator);

  if (estimator->rows_in_sample < max_rows_in_sample) {
    insert_position = estimator->rows_in_sample++;
  } else {
    // The sample is full, use reservoir sampling.
    double rnd = (((double) random()) / ((double) MAX_RANDOM_VALUE));
    if (rnd > 1/(double)estimator->rows_in_table) return;
    insert_position = random() % estimator->rows_in_sample;
  }
  // Finally, extract the item and send it to the sample.
  float* item = palloc(ocl_sizeOfSampleItem(estimator));
  ocl_extractSampleTuple(estimator, rel, new_tuple, item);
  ocl_pushEntryToSampleBufer(estimator, insert_position, item);
  pfree(item);
}

void ocl_notifyEstimatorOfDeletion(Relation rel) {
  if (!ocl_useKDE()) return;
  ocl_estimator_t* estimator = ocl_getEstimator(rel);
  if (estimator == NULL) return;
  // For now, we just use this to update the table counts.
  estimator->rows_in_table--;
}

#endif
