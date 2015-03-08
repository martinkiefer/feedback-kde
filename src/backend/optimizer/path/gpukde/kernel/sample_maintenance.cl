#pragma OPENCL EXTENSION cl_khr_fp64: enable

#ifndef TYPE_DEFINED_
  #if (TYPE == 4)
    typedef float T;
  #elif (TYPE == 8)
    typedef double T;
  #endif
  #define TYPE_DEFINED_
#endif /* TYPE_DEFINED */
   
__kernel void update_sample_quality_metrics(
    __global const T* const local_results,
    __global T* karma,
    __global T* impact,
    unsigned int sample_size,
    T normalization_factor,
    double estimated_selectivity,
    double actual_selectivity,
    double karma_decay,
    double impact_decay
  ) {
  T local_contribution = local_results[get_global_id(0)];

  // Compute the estimate without the current point.
  double adjusted_estimate = estimated_selectivity * sample_size / normalization_factor;
  adjusted_estimate -= local_contribution;
  adjusted_estimate *= normalization_factor / (sample_size - 1);

  // Compute whether this improved or degraded the estimate.
  #ifndef SQUARED_KARMA
  double improvement = fabs(actual_selectivity - adjusted_estimate);
  improvement -= fabs(actual_selectivity - estimated_selectivity);
  // Now compute the karma by normalizing the improvement to [-1,1]
  double local_karma = improvement * sample_size;
  #else
  double improvement = pow(fabs(actual_selectivity - adjusted_estimate),2);
  improvement -= pow(fabs(actual_selectivity - estimated_selectivity),2);
  // Now compute the karma by normalizing the improvement to [-1,1]
  double local_karma = improvement * pow(sample_size,2);
  #endif


  // Now update the array
  karma[get_global_id(0)] *= karma_decay;
  karma[get_global_id(0)] += local_karma;
  
  impact[get_global_id(0)] *= impact_decay;
  impact[get_global_id(0)] += local_contribution;
}

__kernel void get_point_deletion_hitmap(
    __global const T* const data,
    __constant const T* const point,
    __global char* const hitmap
  ) {
  char hit = 1;
  size_t id = get_global_id(0); 
  
  for(unsigned int i = 0; i < D; i++){
    if(data[id*D+i] != point[i]){
      hit = 0;
    }
  }
  
  hitmap[get_global_id(0)] = hit;
}

__kernel void get_karma_threshold_hitmap(
    __global const T* const karma,
    T threshold,
    __global char* const hitmap
  ) {
  hitmap[get_global_id(0)] = karma[get_global_id(0)] < threshold;
}

__kernel void get_karma_threshold_plus_hitmap(
    __global const T* const karma,
    __global const T* const impact,
    __global const T* const local_results,
    T threshold,
    unsigned int rows_in_table,
    double actual_selectivity,
    __global char* const hitmap
  ) {
  // If we are below threshold, we always want to resample
  T local_karma = karma[get_global_id(0)];
  char hit = local_karma < threshold;
  
  //We struggle with points that were deleted but are not queried again.
  //They can be recognised by negative karma and near zero contribution.
  //Better resample those dudes.
  if(local_karma < 0){
    T local_impact = impact[get_global_id(0)];
    hit = local_impact < (get_global_size(0)/rows_in_table) || hit;
  }
  
  //We assume this is a selectivity effectively zero.
  if(actual_selectivity < (0.5)/rows_in_table){
    T local_contribution = local_results[get_global_id(0)];
    //Every sample point with a local contribution > 0.5 is in the query region
    //These points were certainly deleted 
    hit = (local_contribution > 0.5) || hit;
  }
  
  hitmap[get_global_id(0)] = hit;
}
