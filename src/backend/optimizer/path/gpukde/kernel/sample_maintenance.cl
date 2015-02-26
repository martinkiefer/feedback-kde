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
    double karma_decay
  ) {
  T local_contribution = local_results[get_global_id(0)];

  // Compute the estimate without the current point.
  double adjusted_estimate = estimated_selectivity * sample_size / normalization_factor;
  adjusted_estimate -= local_contribution;
  adjusted_estimate *= normalization_factor / (sample_size - 1);

  // Compute whether this improved or degraded the estimate.
  double improvement = fabs(actual_selectivity - adjusted_estimate);
  improvement -= fabs(actual_selectivity - estimated_selectivity);
  
  // Now compute the karma by normalizing the improvement to [-1,1]
  double local_karma = improvement * sample_size;

  // Now update the array
  //karma[get_global_id(0)] *= karma_decay;
  //karma[get_global_id(0)] += (1-karma_decay) * local_karma;
  karma[get_global_id(0)] += local_karma;
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
    const T threshold,
    __global char* const hitmap
  ) {
  hitmap[get_global_id(0)] = karma[get_global_id(0)] < threshold;
}