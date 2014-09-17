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

#include "storage/bufpage.h"
#include "storage/procarray.h"
#include "utils/tqual.h"
#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/xact.h"
#include "storage/bufmgr.h"

#include <math.h>

#ifdef USE_OPENCL

extern ocl_kernel_type_t global_kernel_type;

// GUC configuration variable.
double kde_sample_maintenance_threshold;
double kde_sample_maintenance_karma_decay;
double kde_sample_maintenance_contribution_decay;

bool kde_sample_maintenance_track_impact;
bool kde_sample_maintenance_track_karma;
int kde_sample_maintenance_period;
int kde_sample_maintenance_insert_option;
int kde_sample_maintenance_query_option;

const double sample_match_learning_rate = 0.02f;

static cl_mem getBufferForNextMetric(ocl_estimator_t* estimator) {
  // Switch to the next sample maintenance metric.
  if (kde_sample_maintenance_track_impact &&
      estimator->last_optimized_sample_metric != IMPACT) {
    estimator->last_optimized_sample_metric = IMPACT;
  } else if (kde_sample_maintenance_track_karma &&
      estimator->last_optimized_sample_metric != KARMA) {
    estimator->last_optimized_sample_metric = KARMA;
  }
  // Now return the buffer for the chosen metric.
  if (estimator->last_optimized_sample_metric == IMPACT) {
    return estimator->sample_contribution_buffer;
  } else if (estimator->last_optimized_sample_metric == KARMA) {
    return estimator->sample_karma_buffer;
  }
}

//Convenience method for retrieving the index of the smallest element
static int getMinPenaltyIndex(
    ocl_context_t*  ctxt, ocl_estimator_t* estimator){
  cl_event event;
  unsigned int index;
  kde_float_t val;
  
  // Allocate device memory for indices and values.
  cl_mem min_idx = clCreateBuffer(
          ctxt->context, CL_MEM_READ_WRITE,
          sizeof(unsigned int), NULL, NULL);
  cl_mem min_val = clCreateBuffer(
          ctxt->context, CL_MEM_READ_WRITE,
          sizeof(kde_float_t), NULL, NULL);
  
  // Now fetch the minimum penalty.
  cl_mem target_metric_buffer = getBufferForNextMetric(estimator);
  event = minOfArray(
      target_metric_buffer, estimator->rows_in_sample,
      min_val, min_idx, 0, NULL);
  clEnqueueReadBuffer(
      ctxt->queue, min_idx, CL_TRUE, 0, sizeof(unsigned int),
      &index, 1, &event, NULL);
  clEnqueueReadBuffer(
      ctxt->queue, min_val, CL_TRUE, 0, sizeof(kde_float_t),
      &val, 1, &event, NULL);

  clReleaseEvent(event);
  clReleaseMemObject(min_idx);
  clReleaseMemObject(min_val);

  if (val < 0.1) {
    return index;
  } else {
    return -1;
  }
}

// Helper method to get the minimum index below a certain threshold.
// Returns a NULL pointer if there is none.
static unsigned int *getMinPenaltyIndexBelowThreshold(
    ocl_context_t*  ctxt, ocl_estimator_t* estimator,
    double threshold, cl_event wait_event){
  cl_event event;
  unsigned int *index = palloc(sizeof(unsigned int));
  kde_float_t val;
  
  //Allocate device memory for indices and values.
  cl_mem min_idx = clCreateBuffer(
          ctxt->context, CL_MEM_READ_WRITE,
          sizeof(unsigned int), NULL, NULL);
  cl_mem min_val = clCreateBuffer(
          ctxt->context, CL_MEM_READ_WRITE,
          sizeof(kde_float_t), NULL, NULL);
  
  event = minOfArray(estimator->sample_karma_buffer, estimator->rows_in_sample,
                    min_val, min_idx, 0, wait_event);
  
  clEnqueueReadBuffer(ctxt->queue,min_idx, CL_TRUE, 0,
	                    sizeof(unsigned int), index, 1, &event, NULL);
  clEnqueueReadBuffer(ctxt->queue,min_val, CL_TRUE, 0,
	               sizeof(kde_float_t), &val, 1, &event, NULL);

  clReleaseMemObject(min_idx);
  clReleaseMemObject(min_val);
  
  if(val < threshold) {
    clReleaseEvent(event);
    return index;
  }
  
  pfree(index);
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
  } else if (kde_sample_maintenance_insert_option == RESERVOIR) {
    // The sample is full, use reservoir sampling.
    unsigned int rnd = random() % estimator->rows_in_table;
    if (rnd >= estimator->rows_in_sample) return;
    insert_position = rnd;
  } else if(kde_sample_maintenance_insert_option == RANDOM) {
    //TODO: This is not the best idea.
    //Think of something more flexible and clever.
    // The sample is full, use random sampling
    unsigned int rnd = random() % estimator->rows_in_table;
    if (rnd > estimator->rows_in_sample) return;
    insert_position = getMinPenaltyIndex(ctxt, estimator);
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

static unsigned int min_tuple_size(TupleDesc desc){
  unsigned int min_tuple_size = 0;
  
  int i = 0;
  for (i=0; i < desc->natts; i++) {
    int16 attlen = desc->attrs[i]->attlen;
    
    if(attlen > 0) {
      min_tuple_size += attlen;
    } else {
      //This is one of those filthy variable data types. 
      //There should be at least one byte of overhead.
      //Maybe we can get a better minimum storage requirement bound from somewhere.
      min_tuple_size += 1;
    }
  }
  return min_tuple_size;
}


/*
 * The following method fetches a truly random living tuple from a table
 * and does not need a full table scan.
 * However, it needs an upper bound on the number of tuple per page.
 * Use this method with caution as its performance is highly dependent
 * on that upper bound.
 * 
 * Average number of block reads to access a tuple is:
 * upper_bound / avg. filling degree of pages
 * 
 * And, yes, it will loop forever if there are no alive tuples.
 * 
 */
static HeapTuple sampleTuple(
    Relation rel, BlockNumber blocks, unsigned int max_tuples,
    TransactionId OldestXmin, double *visited_blocks, double* live_rows){
  *visited_blocks = 0;
  *live_rows = 0;
 
  //Very well, he have an upper bound on the number of tuples. So:
  HeapTupleData *used_tuples = (HeapTupleData *) palloc(max_tuples * sizeof(HeapTupleData));
  
  while(1) {
    //Step 3: Select a random page.
    //Can this still be blocks by rounding errors?
    BlockNumber bn = (BlockNumber) (anl_random_fract()*(double) blocks);
    if(bn >= blocks) bn = blocks-1;
  
    ++*visited_blocks;
    OffsetNumber targoffset,maxoffset;
    Page targpage;  
    //Now open the box.
    Buffer targbuffer = ReadBuffer(rel,bn);
    LockBuffer(targbuffer, BUFFER_LOCK_SHARE);
    
    targpage = BufferGetPage(targbuffer);
    maxoffset = PageGetMaxOffsetNumber(targpage);
    //This should never ever happen.
    Assert(maxoffset <= max_tuples);  
    int qualifying_rows = 0;
    
    /* Inner loop over all tuples on the selected page */
    for (targoffset = FirstOffsetNumber; targoffset <= maxoffset; targoffset++) {
      
      ItemId itemid;
      //HeapTupleData targtuple;
      
      itemid = PageGetItemId(targpage, targoffset);
	  
      //This stuff is basically taken from acquire_sample_rows
      if (!ItemIdIsNormal(itemid)) continue;
	  
      ItemPointerSet(&used_tuples[qualifying_rows].t_self, bn, targoffset);

      used_tuples[qualifying_rows].t_data = (HeapTupleHeader) PageGetItem(targpage, itemid);
      used_tuples[qualifying_rows].t_len = ItemIdGetLength(itemid);

      switch (HeapTupleSatisfiesVacuum(
          used_tuples[qualifying_rows].t_data, OldestXmin,targbuffer)) {
        case HEAPTUPLE_LIVE:
          qualifying_rows += 1;
          ++(*live_rows);
          continue;

        case HEAPTUPLE_INSERT_IN_PROGRESS:
          if (TransactionIdIsCurrentTransactionId(
                HeapTupleHeaderGetXmin(used_tuples[qualifying_rows].t_data))) {
            qualifying_rows += 1;
            ++(*live_rows);
            continue;
          }
        case HEAPTUPLE_DELETE_IN_PROGRESS:
          if (!TransactionIdIsCurrentTransactionId(
                HeapTupleHeaderGetUpdateXid(used_tuples[qualifying_rows].t_data))) {
            ++(*live_rows);
          }
        case HEAPTUPLE_DEAD:
        case HEAPTUPLE_RECENTLY_DEAD:
          continue;

        default:
          elog(ERROR, "unexpected HeapTupleSatisfiesVacuum result");
          continue;
      }
    }
    
    // Very well, we know the number of interesting tuples in the page
    // Step 4: Calculate the acceptance rate:
    double acceptance_rate = qualifying_rows/(double) max_tuples;
    if (anl_random_fract() > acceptance_rate){
      UnlockReleaseBuffer(targbuffer);
      continue;
    }
      
    // And we didn't even got rejected, so pick a block.
    int selected_tuple = (int) (anl_random_fract()*(double) (qualifying_rows));
    if (selected_tuple >= qualifying_rows) selected_tuple = qualifying_rows-1;

    HeapTuple tup = heap_copytuple(used_tuples + selected_tuple);
    
    pfree(used_tuples);
    UnlockReleaseBuffer(targbuffer);
    return tup;
  }
}

static int ocl_maxTuplesPerBlock(TupleDesc desc){
  //We will ignore alignment for this calculation.
  //A page consists of (Page Header | N * ItemIds | N * (TupleHeader | Tuple)
  return (int) ((BLCKSZ - SizeOfPageHeaderData) / ((double) min_tuple_size(desc) + sizeof(ItemIdData) + sizeof(HeapTupleHeaderData)));
}

//We will keep this basically consistent with acquire_sample_rows from analyze.c,
int ocl_createSample(Relation rel, HeapTuple *sample,double* estimated_rows,int sample_size){
  
  TransactionId oldestXmin = GetOldestXmin(rel->rd_rel->relisshared, true);
  //Step 1: Get the total number of blocks
  BlockNumber blocks = RelationGetNumberOfBlocks(rel);
  
  //Step 2: Compute an upper bound for the number of tuples in a block
  //2.1: Get the tuple descriptor
  TupleDesc desc = rel->rd_att;
  
  //2.2: Calculate an upper bound based on type information
  int max_tuples = ocl_maxTuplesPerBlock(desc);
  
  double tmp_blocks = 0.0;
  double tmp_tuples = 0.0;
  
  double total_seen_blocks = 0.0;
  double total_seen_tuples = 0.0;

  int i = 0;
  
  for (i = 0; i < sample_size; i++) {
    tmp_blocks = 0.0;
    tmp_tuples = 0.0;
    sample[i] = sampleTuple(
        rel, blocks, max_tuples, oldestXmin, &tmp_blocks, &tmp_tuples);
    total_seen_blocks += tmp_blocks;
    total_seen_tuples += tmp_tuples;
  }
  
  *estimated_rows = vac_estimate_reltuples(
      rel, true,blocks, total_seen_blocks, total_seen_tuples);
  return sample_size;
}

int ocl_isSafeToSample(Relation rel, double total_rows) {
    BlockNumber blocks = RelationGetNumberOfBlocks(rel);
    return blocks == 0 ||
        total_rows == 0 ||
        (ocl_maxTuplesPerBlock(rel->rd_att) / (total_rows / (double) blocks)) > 1.75;
}

void ocl_notifySampleMaintenanceOfSelectivity(
    ocl_estimator_t* estimator, double actual_selectivity) {
  if (estimator == NULL) return;

  estimator->nr_of_estimations++;

  size_t global_size = estimator->rows_in_sample;
  cl_event quality_update_event;

  // Compute the (kernel-specific) normalization factor.
  kde_float_t normalization_factor;
  if (global_kernel_type == EPANECHNIKOV){
    normalization_factor = (kde_float_t) pow(0.75, estimator->nr_of_dimensions);
  } else {
    normalization_factor = (kde_float_t) pow(0.5, estimator->nr_of_dimensions);
  }

  // Schedule the kernel to update the quality factors
  cl_kernel kernel = ocl_getKernel(
      "update_sample_quality_metrics", estimator->nr_of_dimensions);
  ocl_context_t * ctxt = ocl_getContext();
  cl_int err = 0;
  err |= clSetKernelArg(
      kernel, 0, sizeof(cl_mem), &(ctxt->result_buffer));
  err |= clSetKernelArg(
      kernel, 1, sizeof(cl_mem), &(estimator->sample_karma_buffer));
  err |= clSetKernelArg(
      kernel, 2, sizeof(cl_mem), &(estimator->sample_contribution_buffer));
  err |= clSetKernelArg(
      kernel, 3, sizeof(unsigned int), &(estimator->rows_in_sample));
  err |= clSetKernelArg(
      kernel, 4, sizeof(kde_float_t), &(normalization_factor));
  err |= clSetKernelArg(
      kernel, 5, sizeof(double), &(estimator->last_selectivity));
  err |= clSetKernelArg(
      kernel, 6, sizeof(double), &(actual_selectivity));
  err |= clSetKernelArg(
      kernel, 7, sizeof(double), &(kde_sample_maintenance_karma_decay));
  err |= clSetKernelArg(
      kernel, 8, sizeof(double), &(kde_sample_maintenance_contribution_decay));
  err |= clEnqueueNDRangeKernel(
      ctxt->queue, kernel, 1, NULL, &global_size,
      NULL, 0, NULL, &quality_update_event);

  Assert(err == 0);

  if (kde_sample_maintenance_query_option == THRESHOLD) {
    //It might be more efficient to first determine the number of elements to replace
    //and then create a random sample with sufficient size. Maybe later.
    unsigned int *insert_position = getMinPenaltyIndexBelowThreshold(
        ctxt, estimator, kde_sample_maintenance_threshold,
        quality_update_event);
    
    if (insert_position == NULL) return;
    
    //We have got work todo. Get structures to obtain random rows.
    kde_float_t* item = palloc(ocl_sizeOfSampleItem(estimator));
    HeapTuple sample_point;
    
    double total_rows;
    Relation onerel = try_relation_open(estimator->table, ShareUpdateExclusiveLock);
    
    //If the table is in a bad condition, we won't do anything.
    if (insert_position != NULL &&
        ocl_isSafeToSample(onerel,(double) estimator->rows_in_table)){
      pfree(insert_position);
      insert_position = NULL;
    }
    
    while (insert_position != NULL) {
      ocl_createSample(onerel, &sample_point, &total_rows, 1);
      ocl_extractSampleTuple(estimator, onerel, sample_point,item);
      ocl_pushEntryToSampleBufer(estimator, *insert_position, item);
      pfree(insert_position);
      heap_freetuple(sample_point);
      insert_position = getMinPenaltyIndexBelowThreshold(
          ctxt, estimator, kde_sample_maintenance_threshold*-1, NULL);
    }
    pfree(item); 
    relation_close(onerel, ShareUpdateExclusiveLock);
  } else if (kde_sample_maintenance_query_option == PERIODIC &&
      estimator->nr_of_estimations % kde_sample_maintenance_period == 0 ){
    kde_float_t* item;

    HeapTuple sample_point;
    double total_rows;
    
    int insert_position = getMinPenaltyIndex(ctxt, estimator);
    if (insert_position >= 0) {
      Relation onerel = try_relation_open(
          estimator->table, ShareUpdateExclusiveLock);
      // This often prevents Postgres from sampling.
      /*if (ocl_isSafeToSample(onerel,(double) estimator->rows_in_table)) {
        relation_close(onerel, ShareUpdateExclusiveLock);
        return;
      }*/
      item = palloc(ocl_sizeOfSampleItem(estimator));
      ocl_createSample(onerel,&sample_point,&total_rows,1);
      ocl_extractSampleTuple(estimator, onerel, sample_point,item);
      ocl_pushEntryToSampleBufer(estimator, insert_position, item);
      heap_freetuple(sample_point);
      pfree(item);
      relation_close(onerel, ShareUpdateExclusiveLock);
    }
  }
}
#endif
