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

void ocl_allocateSampleMaintenanceBuffers(ocl_estimator_t* estimator) {
  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  ocl_sample_optimization_t* descriptor = calloc(
      1, sizeof(ocl_sample_optimization_t));
  // Allocate two new buffers that we use for storing sample information.
  descriptor->sample_karma_buffer = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->rows_in_sample, NULL, &err);
  Assert(err == CL_SUCCESS);  
  
  descriptor->sample_hitmap = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(char) * estimator->rows_in_sample, NULL, &err);
  Assert(err == CL_SUCCESS);  
  
  descriptor->deleted_point = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * estimator->nr_of_dimensions, NULL, &err);
  Assert(err == CL_SUCCESS);

    // Allocate device memory for indices and values.
  descriptor->min_idx = clCreateBuffer(
          context->context, CL_MEM_READ_WRITE,
          sizeof(unsigned int), NULL, &err);
  Assert(err == CL_SUCCESS);
  
  descriptor->min_val = clCreateBuffer(
          context->context, CL_MEM_READ_WRITE,
          sizeof(kde_float_t), NULL, &err);
  Assert(err == CL_SUCCESS);
  
  // Register the descriptor in the estimator.
  estimator->sample_optimization = descriptor;
}

void ocl_releaseSampleMaintenanceBuffers(ocl_estimator_t* estimator) {
  if (estimator->sample_optimization) {
    cl_int err = CL_SUCCESS;
    ocl_sample_optimization_t* descriptor = estimator->sample_optimization;
    if (descriptor->sample_karma_buffer) {
      err = clReleaseMemObject(descriptor->sample_karma_buffer);
      Assert(err == CL_SUCCESS);
    }
    if (descriptor->sample_hitmap) {
      err = clReleaseMemObject(descriptor->sample_hitmap);
      Assert(err == CL_SUCCESS);
    }
    if (descriptor->deleted_point) {
      err = clReleaseMemObject(descriptor->deleted_point);
      Assert(err == CL_SUCCESS);
    }
    if (descriptor->min_idx) {
      err = clReleaseMemObject(descriptor->min_idx);
      Assert(err == CL_SUCCESS);
    }
    if (descriptor->min_val) {
      err = clReleaseMemObject(descriptor->min_val);
      Assert(err == CL_SUCCESS);
    }    
    
    free(estimator->sample_optimization);
  }
}

// GUC configuration variable.
double kde_sample_maintenance_threshold;
double kde_sample_maintenance_karma_decay;

int kde_sample_maintenance_period;
int kde_sample_maintenance_option;

//Convenience method for retrieving the index of the smallest element
static int getMinPenaltyIndex(
    ocl_context_t*  ctxt, ocl_estimator_t* estimator){
  cl_event event;
  unsigned int index;
  cl_int err = CL_SUCCESS;
    
  // Now fetch the minimum penalty.
  event = minOfArray(
      estimator->sample_optimization->sample_karma_buffer, estimator->rows_in_sample,
      estimator->sample_optimization->min_val, estimator->sample_optimization->min_idx, 0, NULL);
  
  err |= clEnqueueReadBuffer(
      ctxt->queue, estimator->sample_optimization->min_idx, CL_TRUE, 0, sizeof(unsigned int),
      &index, 1, &event, NULL);
  estimator->stats->maintenance_transfer_to_host++;
  Assert(err == CL_SUCCESS);

  err = clReleaseEvent(event);
  Assert(err == CL_SUCCESS);

  return index;
}

// Helper method to get the minimum index below a certain threshold.
// Returns a NULL pointer if there is none.
static unsigned int *getMinPenaltyIndexBelowThreshold(
    ocl_context_t*  ctxt, ocl_estimator_t* estimator,
    double threshold, cl_event wait_event){
  cl_event event;
  cl_int err = CL_SUCCESS;
  unsigned int *index = palloc(sizeof(unsigned int));
  kde_float_t val;
  
  //Allocate device memory for indices and values.
  cl_mem min_idx = clCreateBuffer(
          ctxt->context, CL_MEM_READ_WRITE,
          sizeof(unsigned int), NULL, &err);
  Assert(err == CL_SUCCESS);
  cl_mem min_val = clCreateBuffer(
          ctxt->context, CL_MEM_READ_WRITE,
          sizeof(kde_float_t), NULL, &err);
  Assert(err == CL_SUCCESS);
  
  event = minOfArray(
      estimator->sample_optimization->sample_karma_buffer,
      estimator->rows_in_sample, min_val, min_idx, 0, wait_event);
  
  err |= clEnqueueReadBuffer(
      ctxt->queue,min_idx, CL_TRUE, 0, sizeof(unsigned int),
      index, 1, &event, NULL);
  estimator->stats->maintenance_transfer_to_host++;
  err |= clEnqueueReadBuffer(
      ctxt->queue,min_val, CL_TRUE, 0, sizeof(kde_float_t),
      &val, 1, &event, NULL);
  estimator->stats->maintenance_transfer_to_host++;
  Assert(err == CL_SUCCESS);
  
  err |= clReleaseMemObject(min_idx);
  err |= clReleaseMemObject(min_val);
  
  if(val < threshold) {
    err = clReleaseEvent(event);
    Assert(err == CL_SUCCESS);
    return index;
  }
  
  pfree(index);
  err = clReleaseEvent(event);
  Assert(err == CL_SUCCESS);
  
  return NULL; 
}

//Efficient implementation of drawing from a binomial distribution for p*n small.
static int getBinomial(int n, double p) {
   double log_q = log(1.0 - p);
   //Bad things happen if p equals 1.
   if(log_q == -INFINITY) return n;
   int x = 0;
   double sum = 0;
   for(;;) {
      sum += log((double) random() / (double) (RAND_MAX - 1) ) / (n - x);
      if(sum < log_q) {
         return x;
      }
      x++;
   }
}

static void trigger_periodic_random_replacement(ocl_estimator_t* estimator){
  if (kde_sample_maintenance_option == PRR &&
      (estimator->stats->nr_of_insertions + estimator->stats->nr_of_deletions) % kde_sample_maintenance_period == 0 ){
    kde_float_t* item;

    HeapTuple sample_point;
    double total_rows;
    
    int insert_position = random() % estimator->rows_in_sample;
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
      estimator->stats->maintenance_transfer_to_device++;
      
      
      heap_freetuple(sample_point);
      pfree(item);
      relation_close(onerel, ShareUpdateExclusiveLock);
    }
  }
}

void ocl_notifySampleMaintenanceOfInsertion(Relation rel, HeapTuple new_tuple) {
  // Check whether we have a table for this relation.
  ocl_estimator_t* estimator = ocl_getEstimator(rel->rd_id);
  int i = 0;
  
  if (estimator == NULL) return;
  estimator->rows_in_table++;
  estimator->stats->nr_of_insertions++;
  
  if (! kde_sample_maintenance_option) return;
  int insert_position = -1;
  
  // First, check whether we still have size in the sample.
  if (estimator->rows_in_sample < ocl_maxRowsInSample(estimator)) {
    insert_position = estimator->rows_in_sample++;
  } else if (kde_sample_maintenance_option == CAR) {
    // The sample is full, use CAR.
    int replacements = getBinomial(estimator->rows_in_sample, 1.0 / estimator->rows_in_table);
    if(replacements > 0){
      kde_float_t* item = palloc(ocl_sizeOfSampleItem(estimator));
      ocl_extractSampleTuple(estimator, rel, new_tuple, item);
      for(i=0; i < replacements; i++){
	insert_position = random() % estimator->rows_in_sample;
	ocl_pushEntryToSampleBufer(estimator, insert_position, item);
	estimator->stats->maintenance_transfer_to_device++;
      }
      pfree(item);
    }
  }
  else if(kde_sample_maintenance_option == PRR){
    trigger_periodic_random_replacement(estimator);
  }  
}

void ocl_notifySampleMaintenanceOfDeletion(Relation rel, ItemPointer tupleid) {
  ocl_estimator_t* estimator = ocl_getEstimator(rel->rd_id);
  if (estimator == NULL) return;
  // For now, we just use this to update the table counts.
  estimator->rows_in_table--;
  estimator->stats->nr_of_deletions++;
  if(kde_sample_maintenance_option == PRR){
    trigger_periodic_random_replacement(estimator);
  }
  else if(kde_sample_maintenance_option == CAR){
    ocl_context_t* ctxt = ocl_getContext();
    HeapTupleData deltuple;
    deltuple.t_self = *tupleid;
    Buffer		delbuffer;
    int err = 0;
    
    kde_float_t* tuple_buffer = (kde_float_t *) palloc(estimator->nr_of_dimensions * (sizeof(kde_float_t)));
    
    heap_fetch(rel, SnapshotAny,&deltuple, &delbuffer, false, NULL);
    ocl_extractSampleTuple(estimator,rel,&deltuple,tuple_buffer);
    Assert(BufferIsValid(delbuffer));
    ReleaseBuffer(delbuffer);
    
    err |= clEnqueueWriteBuffer(
        ctxt->queue, estimator->sample_optimization->deleted_point, CL_TRUE, 0,
        ocl_sizeOfSampleItem(estimator),
        tuple_buffer, 0, NULL, NULL);
    estimator->stats->maintenance_transfer_to_device++;   
    Assert(err == CL_SUCCESS);
    pfree(tuple_buffer); 
    unsigned int i = 0;
    cl_event hitmap_event;
    size_t global_size = estimator->rows_in_sample;
    char* hitmap = (char*) palloc(global_size*sizeof(char));
    
    cl_kernel kernel = ocl_getKernel(
      "get_point_deletion_hitmap", estimator->nr_of_dimensions);
    err |= clSetKernelArg(
      kernel, 0, sizeof(cl_mem), &(estimator->sample_buffer));
    err |= clSetKernelArg(
      kernel, 1, sizeof(cl_mem), &(estimator->sample_optimization->deleted_point));
    err |= clSetKernelArg(
      kernel, 2, sizeof(cl_mem), &(estimator->sample_optimization->sample_hitmap));   
    Assert(err == CL_SUCCESS);
    
    err = clEnqueueNDRangeKernel(
      ctxt->queue, kernel, 1, NULL, &global_size,
      NULL, 0, NULL, &hitmap_event);
    Assert(err == CL_SUCCESS);
    
    err = clEnqueueReadBuffer(
      ctxt->queue, estimator->sample_optimization->sample_hitmap, CL_TRUE, 0, sizeof(char) * global_size,
      hitmap, 1, &hitmap_event, NULL);
    estimator->stats->maintenance_transfer_to_host++;
    Assert(err == CL_SUCCESS);
    
    //We have got work todo. Get structures to obtain random rows.
    kde_float_t* item = palloc(ocl_sizeOfSampleItem(estimator));
    HeapTuple sample_point;

    double total_rows;
    
    for(i=0; i < estimator->rows_in_sample; i++){
      if(hitmap[i]){
	ocl_createSample(rel, &sample_point, &total_rows, 1);
	ocl_extractSampleTuple(estimator, rel, sample_point,item);
	ocl_pushEntryToSampleBufer(estimator, i, item);
	estimator->stats->maintenance_transfer_to_device++;
	heap_freetuple(sample_point);	
      }
    }
    pfree(item); 
    pfree(hitmap);
  }
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
    //This should never ever happen otherwise we can't guarantee uniform sampling.
    Assert(qualifying_rows <= max_tuples);
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
  
  *estimated_rows = blocks * total_seen_tuples/total_seen_blocks;
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
  estimator->stats->nr_of_estimations++;

  //PRR and CAR do not need the karma metric.
  if (kde_sample_maintenance_option != TKR && kde_sample_maintenance_option != PKR) return;
  
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
      kernel, 0, sizeof(cl_mem), &(estimator->local_results_buffer));
  err |= clSetKernelArg(
      kernel, 1, sizeof(cl_mem), &(estimator->sample_optimization->sample_karma_buffer));
  err |= clSetKernelArg(
      kernel, 2, sizeof(unsigned int), &(estimator->rows_in_sample));
  err |= clSetKernelArg(
      kernel, 3, sizeof(kde_float_t), &(normalization_factor));
  err |= clSetKernelArg(
      kernel, 4, sizeof(double), &(estimator->last_selectivity));
  err |= clSetKernelArg(
      kernel, 5, sizeof(double), &(actual_selectivity));
  err |= clSetKernelArg(
      kernel, 6, sizeof(double), &(kde_sample_maintenance_karma_decay));
  Assert(err == CL_SUCCESS);
  
  err = clEnqueueNDRangeKernel(
      ctxt->queue, kernel, 1, NULL, &global_size,
      NULL, 0, NULL, &quality_update_event);
  Assert(err == CL_SUCCESS);

  if (kde_sample_maintenance_option == TKR) {
    //It might be more efficient to first determine the number of elements to replace
    //and then create a random sample with sufficient size. Maybe later.
    unsigned int i = 0;
    cl_event hitmap_event;
    char* hitmap = (char*) palloc(global_size*sizeof(char));
    
    cl_kernel kernel = ocl_getKernel(
      "get_karma_threshold_hitmap", estimator->nr_of_dimensions);
    err |= clSetKernelArg(
      kernel, 0, sizeof(cl_mem), &(estimator->sample_optimization->sample_karma_buffer));
    err |= clSetKernelArg(
      kernel, 1, sizeof(kde_float_t), &kde_sample_maintenance_threshold);
    err |= clSetKernelArg(
      kernel, 2, sizeof(cl_mem), &(estimator->sample_optimization->sample_hitmap));    
    err = clEnqueueNDRangeKernel(
      ctxt->queue, kernel, 1, NULL, &global_size,
      NULL, 1, &quality_update_event, &hitmap_event);
    Assert(err == CL_SUCCESS);
    
    err |= clEnqueueReadBuffer(
      ctxt->queue, estimator->sample_optimization->sample_hitmap, CL_TRUE, 0, sizeof(char) * global_size,
      hitmap, 1, &hitmap_event, NULL);
    estimator->stats->maintenance_transfer_to_host++;
    
        //We have got work todo. Get structures to obtain random rows.
    kde_float_t* item = palloc(ocl_sizeOfSampleItem(estimator));
    HeapTuple sample_point;
    
    double total_rows;
    Relation onerel = try_relation_open(estimator->table, ShareUpdateExclusiveLock);
    
    for(i=0; i < global_size; i++){
      if(hitmap[i]){
	ocl_createSample(onerel, &sample_point, &total_rows, 1);
	ocl_extractSampleTuple(estimator, onerel, sample_point,item);
	ocl_pushEntryToSampleBufer(estimator, i, item);
	estimator->stats->maintenance_transfer_to_device++;
	heap_freetuple(sample_point);	
      }
    }  
    pfree(item); 
    relation_close(onerel, ShareUpdateExclusiveLock);
  }
  else if (kde_sample_maintenance_option == PKR &&
      estimator->stats->nr_of_estimations % kde_sample_maintenance_period == 0 ){
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
      estimator->stats->maintenance_transfer_to_device++;
      heap_freetuple(sample_point);
      pfree(item);
      relation_close(onerel, ShareUpdateExclusiveLock);
    }
  }
}  
#endif
