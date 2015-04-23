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
#include <sys/time.h>

#ifdef USE_OPENCL

extern ocl_kernel_type_t global_kernel_type;

#define CHECK_BIT(var,pos) ((var) & (1<<(pos)))
#define SET_BIT(var,pos) (var |= (1<<(pos)))

// GUC configuration variable.
double kde_sample_maintenance_threshold;
double kde_sample_maintenance_karma_limit;

int kde_sample_maintenance_period;
int kde_sample_maintenance_option;

static void ocl_prepareDeletionDescriptor(ocl_estimator_t* estimator, ocl_sample_optimization_t* sample_optimization){  
  ocl_deletion_descriptor_t * desc = calloc(1, sizeof(ocl_deletion_descriptor_t));
  ocl_context_t* ctxt = ocl_getContext();
  
  int err = 0;
  size_t global_size = estimator->rows_in_sample;
  
  desc->deletion_kernel = ocl_getKernel(
    "get_point_deletion_bitmap", estimator->nr_of_dimensions);
  
  size_t max_size;
  err |= clGetKernelWorkGroupInfo(
        desc->deletion_kernel, ctxt->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(size_t), &(desc->local_size), NULL);
  Assert(err == CL_SUCCESS);
  err |= clGetKernelWorkGroupInfo(
        desc->deletion_kernel, ctxt->device, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(size_t), &(max_size), NULL);
  Assert(err == CL_SUCCESS);

  //Workgroup size has to be at least and a multiple of 8
  if(global_size % desc->local_size != 0 || desc->local_size < 8){
	desc->local_size = 8;
  }

  err |= clSetKernelArg(
    desc->deletion_kernel, 0, sizeof(cl_mem), &(estimator->sample_buffer));
  err |= clSetKernelArg(
    desc->deletion_kernel, 1, sizeof(cl_mem), &(sample_optimization->deleted_point));
  err |= clSetKernelArg(
    desc->deletion_kernel, 2, sizeof(kde_float_t)*estimator->nr_of_dimensions, NULL);
  err |= clSetKernelArg(
    desc->deletion_kernel, 3, sizeof(unsigned int)*desc->local_size, NULL);  
  err |= clSetKernelArg(
    desc->deletion_kernel, 4, sizeof(cl_mem), &(sample_optimization->sample_hitmap));   
  
  Assert(err == CL_SUCCESS);

  sample_optimization->del_desc = desc;
}

static void ocl_releaseDeletionDescriptor(ocl_deletion_descriptor_t* del_desc){
  cl_int err = 0;  
  
  err = clReleaseKernel(del_desc->deletion_kernel);
  Assert(err == CL_SUCCESS);
  
  free(del_desc);
}

static void ocl_prepareTkrDescriptor(ocl_estimator_t* estimator, ocl_sample_optimization_t* sample_optimization){  
  ocl_tkr_descriptor_t * desc = calloc(1, sizeof(ocl_tkr_descriptor_t));
  ocl_context_t* ctxt = ocl_getContext();
  
  cl_int err = 0;
  size_t global_size = estimator->rows_in_sample;
  
  desc->tkr_kernel = ocl_getKernel(
    "get_karma_threshold_bitmap", estimator->nr_of_dimensions);
  
  size_t max_size;
  err |= clGetKernelWorkGroupInfo(
        desc->tkr_kernel, ctxt->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(size_t), &(desc->local_size), NULL);
  Assert(err == CL_SUCCESS);
  err |= clGetKernelWorkGroupInfo(
        desc->tkr_kernel, ctxt->device, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(size_t), &(max_size), NULL);
  Assert(err == CL_SUCCESS);

  //Workgroup size has to be at least and a multiple of 8
  if(global_size % desc->local_size != 0 || desc->local_size < 8){
	desc->local_size = 8;
  }

  err |= clSetKernelArg(
    desc->tkr_kernel, 0, sizeof(cl_mem), &(sample_optimization->sample_karma_buffer));
  err |= clSetKernelArg(
    desc->tkr_kernel, 1, sizeof(cl_mem), &(estimator->local_results_buffer));
  err |= clSetKernelArg(
    desc->tkr_kernel, 2, sizeof(cl_mem), &(estimator->input_buffer));
  err |= clSetKernelArg(
    desc->tkr_kernel, 3, sizeof(cl_mem), &(estimator->bandwidth_buffer));
    err |= clSetKernelArg(
    desc->tkr_kernel, 4, sizeof(unsigned int)*desc->local_size, NULL);
  err |= clSetKernelArg(
    desc->tkr_kernel, 5, sizeof(kde_float_t), &kde_sample_maintenance_threshold);
  err |= clSetKernelArg(
    desc->tkr_kernel, 7, sizeof(cl_mem), &(sample_optimization->sample_hitmap));
  
  Assert(err == CL_SUCCESS);

  sample_optimization->tkr_desc = desc;
}

static void ocl_releaseTkrDescriptor(ocl_tkr_descriptor_t* desc){
  cl_int err = 0;  
  
  err = clReleaseKernel(desc->tkr_kernel);
  Assert(err == CL_SUCCESS);
  
  free(desc);
}

static void setActualSelectivity(ocl_tkr_descriptor_t * desc, double actual_selectivity){
  cl_int err = 0;
  err = clSetKernelArg(
    desc->tkr_kernel, 6, sizeof(kde_float_t), &actual_selectivity);
  Assert(err == CL_SUCCESS);
}

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
      sizeof(unsigned char) * (estimator->rows_in_sample/8), NULL, &err);
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
  
  if(kde_sample_maintenance_option == CAR){
    ocl_prepareDeletionDescriptor(estimator, descriptor);
  }
  else if(kde_sample_maintenance_option == TKR) {
    ocl_prepareTkrDescriptor(estimator, descriptor);
  }
  
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
    if(descriptor->del_desc){ 
      ocl_releaseDeletionDescriptor(descriptor->del_desc);
    }
    if(descriptor->tkr_desc){ 
      ocl_releaseTkrDescriptor(descriptor->tkr_desc);
    }
    free(estimator->sample_optimization);
  }
}

//Convenience method for retrieving the index of the smallest element
static int getMinPenaltyIndex(
    ocl_context_t*  ctxt, ocl_estimator_t* estimator, cl_event wait_event){
  cl_event event;
  unsigned int index;
  struct timeval tvBegin, tvEnd;
  cl_int err = CL_SUCCESS;
    
  // Now fetch the minimum penalty.
  event = minOfArray(
      estimator->sample_optimization->sample_karma_buffer, estimator->rows_in_sample,
      estimator->sample_optimization->min_val, estimator->sample_optimization->min_idx, 0, wait_event);
  
  err = clWaitForEvents(1,&event);
  Assert(err == CL_SUCCESS);
  gettimeofday(&tvBegin,NULL);
  err |= clEnqueueReadBuffer(
      ctxt->queue, estimator->sample_optimization->min_idx, CL_TRUE, 0, sizeof(unsigned int),
      &index, 1, &event, NULL);
  gettimeofday(&tvEnd,NULL);
  estimator->stats->maintenance_transfer_time += (tvEnd.tv_sec - tvBegin.tv_sec) * 1000 * 1000;
  estimator->stats->maintenance_transfer_time += (tvEnd.tv_usec - tvBegin.tv_usec);
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

//Efficient implementation of drawing n random array indices without replacements
//n: maximum array index + 1
//m: Number of elements to be sampled
//returns: Bitmask with bits at drawn index sets
static unsigned char* floydSampling(unsigned char* map, int n, int m){
  int j = (n - m + 1);
  for(; j <= n; j++){
    int t = (random() % j) + 1;    
    if(CHECK_BIT(map[(t-1) / 8], (t-1) % 8)){
      SET_BIT(map[(j-1) / 8],(j-1) % 8);
    }
    else {
      SET_BIT(map[(t-1) / 8],(t-1) % 8);
    }
  }
  return map;
}  

static void trigger_periodic_random_replacement(ocl_estimator_t* estimator){
  if (kde_sample_maintenance_option == PRR &&
      (estimator->stats->nr_of_estimations % kde_sample_maintenance_period) == 0 ){
    kde_float_t* item;

    HeapTuple sample_point;
    double total_rows;
  
    struct timeval tvBegin, tvEnd;
    
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
      gettimeofday(&tvBegin,NULL);
      ocl_pushEntryToSampleBufer(estimator, insert_position, item);
      gettimeofday(&tvEnd,NULL);
      estimator->stats->maintenance_transfer_time += (tvEnd.tv_sec - tvBegin.tv_sec) * 1000 * 1000;
      estimator->stats->maintenance_transfer_time += (tvEnd.tv_usec - tvBegin.tv_usec);
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
  if (estimator == NULL) return;

  estimator->rows_in_table++;
  estimator->stats->nr_of_insertions++;
  
  if (kde_sample_maintenance_option != CAR) return;
  struct timeval tvBegin, tvEnd;

  // First, check whether we still have size in the sample.
  if (kde_sample_maintenance_option == CAR) {
    // The sample is full, use CAR.
    int replacements = getBinomial(estimator->rows_in_sample, 1.0 / estimator->rows_in_table);
    if (replacements > 0) {
      size_t map_size = sizeof(unsigned char)*((estimator->rows_in_sample+8-1)/8);
      kde_float_t* item = palloc(ocl_sizeOfSampleItem(estimator));
      unsigned char* index_map = (unsigned char*) palloc0(map_size);
      
      ocl_extractSampleTuple(estimator, rel, new_tuple, item);
     
      index_map = floydSampling(index_map,estimator->rows_in_sample, replacements);
      int i = 0;
      for (; i < map_size; i++) {
	int j = 0;
	while(index_map[i]){
	  if(index_map[i] & 1){
	    gettimeofday(&tvBegin,NULL);
	    ocl_pushEntryToSampleBufer(estimator, i * 8 + j, item);
	    gettimeofday(&tvEnd,NULL);
	    estimator->stats->maintenance_transfer_time += (tvEnd.tv_sec - tvBegin.tv_sec) * 1000 * 1000;
	    estimator->stats->maintenance_transfer_time += (tvEnd.tv_usec - tvBegin.tv_usec);
	    estimator->stats->maintenance_transfer_to_device++;
	  }
	  index_map[i] = index_map[i] >> 1;
	  j++;
	}
      }
      pfree(item);
      pfree(index_map);
    }
  }
}

void ocl_notifySampleMaintenanceOfDeletion(Relation rel, ItemPointer tupleid) {
  ocl_estimator_t* estimator = ocl_getEstimator(rel->rd_id);
  if (estimator == NULL) return;
  struct timeval tvBegin, tvEnd;

  // For now, we just use this to update the table counts.
  estimator->rows_in_table--;
  estimator->stats->nr_of_deletions++;

  if(kde_sample_maintenance_option == CAR){
    ocl_context_t* ctxt = ocl_getContext();
    size_t global_size = estimator->rows_in_sample;
    size_t bitmap_size = estimator->rows_in_sample / 8;
    HeapTupleData deltuple;
    deltuple.t_self = *tupleid;
    Buffer		delbuffer;
    int err = 0;
    
    kde_float_t* tuple_buffer = (kde_float_t *) palloc(estimator->nr_of_dimensions * (sizeof(kde_float_t)));
    
    heap_fetch(rel, SnapshotAny,&deltuple, &delbuffer, false, NULL);
    ocl_extractSampleTuple(estimator,rel,&deltuple,tuple_buffer);
    Assert(BufferIsValid(delbuffer));
    ReleaseBuffer(delbuffer);
     
    unsigned int i = 0;
    cl_event hitmap_event;
    
    
    unsigned char* hitmap = (unsigned char*) palloc(bitmap_size*sizeof(unsigned char));
    
    gettimeofday(&tvBegin,NULL);
    err |= clEnqueueWriteBuffer(
      ctxt->queue, estimator->sample_optimization->deleted_point, CL_TRUE, 0,
      ocl_sizeOfSampleItem(estimator),
      tuple_buffer, 0, NULL, NULL);
    gettimeofday(&tvEnd,NULL);
    estimator->stats->maintenance_transfer_time += (tvEnd.tv_sec - tvBegin.tv_sec) * 1000 * 1000;
    estimator->stats->maintenance_transfer_time += (tvEnd.tv_usec - tvBegin.tv_usec);
    estimator->stats->maintenance_transfer_to_device++;
    Assert(err == CL_SUCCESS);
    
    err = clEnqueueNDRangeKernel(
      ctxt->queue, estimator->sample_optimization->del_desc->deletion_kernel, 1, NULL, &global_size,
      &(estimator->sample_optimization->del_desc->local_size), 0, NULL, &hitmap_event);
    Assert(err == CL_SUCCESS);
    
    err = clWaitForEvents(1,&hitmap_event);
    Assert(err == CL_SUCCESS);
    gettimeofday(&tvBegin,NULL);
    err = clEnqueueReadBuffer(
      ctxt->queue, estimator->sample_optimization->sample_hitmap, CL_TRUE, 0, sizeof(char) * bitmap_size,
      hitmap, 1, &hitmap_event, NULL);
    gettimeofday(&tvEnd,NULL);
    estimator->stats->maintenance_transfer_time += (tvEnd.tv_sec - tvBegin.tv_sec) * 1000 * 1000;
    estimator->stats->maintenance_transfer_time += (tvEnd.tv_usec - tvBegin.tv_usec);
    estimator->stats->maintenance_transfer_to_host++;
    Assert(err == CL_SUCCESS);
    err = clReleaseEvent(hitmap_event);
    Assert(err == CL_SUCCESS);

    //We have got work todo. Get structures to obtain random rows.
    kde_float_t* item = palloc(ocl_sizeOfSampleItem(estimator));
    HeapTuple sample_point;

    double total_rows;
    
    for(i=0; i < bitmap_size; i++){
      int j=0;
      while(hitmap[i]){
	if(hitmap[i] & 1){
	  ocl_createSample(rel, &sample_point, &total_rows, 1);
	  ocl_extractSampleTuple(estimator, rel, sample_point,item);
	  gettimeofday(&tvBegin,NULL);
	  ocl_pushEntryToSampleBufer(estimator, i*8+j, item);
	  gettimeofday(&tvEnd,NULL);
	  estimator->stats->maintenance_transfer_time += (tvEnd.tv_sec - tvBegin.tv_sec) * 1000 * 1000;
	  estimator->stats->maintenance_transfer_time += (tvEnd.tv_usec - tvBegin.tv_usec);
	  estimator->stats->maintenance_transfer_to_device++;
	  heap_freetuple(sample_point);
	}
        j++;
	hitmap[i] = hitmap[i] >> 1; 
      }
    }
    pfree(item); 
    pfree(hitmap);
    pfree(tuple_buffer);
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

  if(kde_sample_maintenance_option == PRR){
    trigger_periodic_random_replacement(estimator);
  }
  
  // PRR and CAR do not need the karma metric.
  if (kde_sample_maintenance_option != TKR 
      && kde_sample_maintenance_option != PKR) {
    return;
  }

  struct timeval tvBegin, tvEnd;
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
      kernel, 6, sizeof(double), &(kde_sample_maintenance_karma_limit));
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
    int bitmap_size = global_size / 8;
    unsigned char* hitmap = (unsigned char*) palloc(bitmap_size*sizeof(unsigned char));
    
    setActualSelectivity(estimator->sample_optimization->tkr_desc,actual_selectivity);
    err = clEnqueueNDRangeKernel(
      ctxt->queue, estimator->sample_optimization->tkr_desc->tkr_kernel, 1, NULL, &global_size,
      &(estimator->sample_optimization->tkr_desc->local_size), 1, &quality_update_event, &hitmap_event);
    Assert(err == CL_SUCCESS);
    
    err = clWaitForEvents(1,&hitmap_event);
    Assert(err == CL_SUCCESS);
    gettimeofday(&tvBegin,NULL);
    err |= clEnqueueReadBuffer(
      ctxt->queue, estimator->sample_optimization->sample_hitmap, CL_TRUE, 0, sizeof(char) * bitmap_size,
      hitmap, 1, &hitmap_event, NULL);
    gettimeofday(&tvEnd,NULL);
    estimator->stats->maintenance_transfer_time += (tvEnd.tv_sec - tvBegin.tv_sec) * 1000 * 1000;
    estimator->stats->maintenance_transfer_time += (tvEnd.tv_usec - tvBegin.tv_usec);
    estimator->stats->maintenance_transfer_to_host++;
    Assert(err == CL_SUCCESS);
    err = clReleaseEvent(hitmap_event);
    Assert(err == CL_SUCCESS);
    err = clReleaseEvent(quality_update_event);
    Assert(err == CL_SUCCESS);
    
        //We have got work todo. Get structures to obtain random rows.
    kde_float_t* item = palloc(ocl_sizeOfSampleItem(estimator));
    HeapTuple sample_point;
    
    double total_rows;
    Relation rel = try_relation_open(estimator->table, ShareUpdateExclusiveLock);
    
    for(i=0; i < bitmap_size; i++){
      int j=0;
      while(hitmap[i]){
	if(hitmap[i] & 1){
	  ocl_createSample(rel, &sample_point, &total_rows, 1);
	  ocl_extractSampleTuple(estimator, rel, sample_point,item);
	  gettimeofday(&tvBegin,NULL);
	  ocl_pushEntryToSampleBufer(estimator, i*8+j, item);
	  gettimeofday(&tvEnd,NULL);
	  estimator->stats->maintenance_transfer_time += (tvEnd.tv_sec - tvBegin.tv_sec) * 1000 * 1000;
	  estimator->stats->maintenance_transfer_time += (tvEnd.tv_usec - tvBegin.tv_usec);
	  estimator->stats->maintenance_transfer_to_device++;
	  heap_freetuple(sample_point);
	}
        j++;
	hitmap[i] = hitmap[i] >> 1; 
      }
    } 
    pfree(item); 
    relation_close(rel, ShareUpdateExclusiveLock);
  }
  else if (kde_sample_maintenance_option == PKR){
    kde_float_t* item;

    HeapTuple sample_point;
    double total_rows;
    
    if(estimator->stats->nr_of_estimations % kde_sample_maintenance_period != 0){
      err = clReleaseEvent(quality_update_event);
      Assert(err == CL_SUCCESS);
      return;
    }
    
    int insert_position = getMinPenaltyIndex(ctxt, estimator,quality_update_event);
    clReleaseEvent(quality_update_event);
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
      gettimeofday(&tvBegin,NULL);
      ocl_pushEntryToSampleBufer(estimator, insert_position, item);
      gettimeofday(&tvEnd,NULL);
      estimator->stats->maintenance_transfer_time += (tvEnd.tv_sec - tvBegin.tv_sec) * 1000 * 1000;
      estimator->stats->maintenance_transfer_time += (tvEnd.tv_usec - tvBegin.tv_usec);
      estimator->stats->maintenance_transfer_to_device += 2;
      heap_freetuple(sample_point);
      pfree(item);
      relation_close(onerel, ShareUpdateExclusiveLock);
    }
  }
}  
#endif
