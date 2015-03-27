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
    unsigned int sample_size,
    T normalization_factor,
    double estimated_selectivity,
    double actual_selectivity,
    double karma_limit
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
  double improvement = pow(fabs(actual_selectivity - adjusted_estimate),2.0);
  improvement -= pow(fabs(actual_selectivity - estimated_selectivity),2.0);
  // Now compute the karma by normalizing the improvement to [-1,1]
  double local_karma = improvement * pow(sample_size,2.0);
#endif

  // Now update the array
  //karma[get_global_id(0)] *= karma_decay;
  karma[get_global_id(0)] += local_karma;
  karma[get_global_id(0)] = fmin(karma[get_global_id(0)], karma_limit);   
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

__kernel void get_point_deletion_bitmap(
    __global const T* const data,
    __constant const T* const point,
    __global unsigned char* const hitmap
  ) {
  unsigned char result = 0;
  
  size_t id = get_global_id(0); 
  for(unsigned int j = 0; j < 8; j++){ 
    char hit = 1;
    for(unsigned int i = 0; i < D; i++){
      if(data[8*id*D+j*D+i] != point[i]){
	hit = 0;
      }
    }
    if(hit){
      result |= 1 << j;
    }
  }
  
  hitmap[get_global_id(0)] = result;
}

__kernel void get_karma_threshold_hitmap(
    __global const T* const karma,
    __global const T* const local_results,
    T threshold,
    double actual_selectivity,
    __global char* const hitmap
  ) {
  // If we are below threshold, we always want to resample
  T local_karma = karma[get_global_id(0)];
  char hit = local_karma < threshold;
  
  if(actual_selectivity == 0.0){
    T local_contribution = local_results[get_global_id(0)];
    //Every sample point with a local contribution > 0.5 is in the query region
    //These points were certainly deleted 
    hit = (local_contribution > 0.5) || hit;
  }
  
  hitmap[get_global_id(0)] = hit;
}

__kernel void get_karma_threshold_bitmap(
    __global const T* const karma,
    __global const T* const local_results,
    T threshold,
    double actual_selectivity,
    __global char* const hitmap
  ) {
  unsigned char result = 0;
  for(unsigned int i = 0; i < 8; i++){
  // If we are below threshold, we always want to resample
    T local_karma = karma[get_global_id(0)*8+i];
    char hit = local_karma < threshold;
 
    if(actual_selectivity == 0.0){
      T local_contribution = local_results[get_global_id(0)*8+i];
      //Every sample point with a local contribution > 0.5 is in the query region
      //These points were certainly deleted 
      hit = (local_contribution > 0.5) || hit;
    }
    if(hit){
      result |= 1 << i;
    }
  }
  
  hitmap[get_global_id(0)] = result;
}
