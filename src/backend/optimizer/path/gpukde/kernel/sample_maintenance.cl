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

__kernel void get_point_deletion_bitmap(
    __global const T* const data,
    __constant const T* const point,
    __local T* lp,
    __local unsigned int* hit,
    __global unsigned char* const hitmap
  ) {
  unsigned char result = 0;

  if(get_local_id(0) < D){
    lp[get_local_id(0)] = point[get_local_id(0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  
  
  unsigned int temp = 1;
  for(unsigned int i = 0; i < D; i++){
    if(data[D*get_global_id(0) +i] != lp[i]){
      temp = 0;
    }
  }
  hit[get_local_id(0)] = temp << get_local_id(0) % 8;
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if(get_local_id(0) % 8 < 4){
    hit[get_local_id(0)] |= hit[get_local_id(0)+4];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  
  if(get_local_id(0) % 8 < 2){
    hit[get_local_id(0)] |= hit[get_local_id(0)+2];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  
  if(get_local_id(0) % 8 < 1){
    hitmap[get_group_id(0)*get_local_size(0)/8 + get_local_id(0) / 8] = (hit[get_local_id(0)] | hit[get_local_id(0)+1]) & 0xFF;
  }
}

__kernel void get_karma_threshold_bitmap(
    __global const T* const karma,
    __global const T* const local_results,
    __global const T* const query,
    __global const T* const bandwidth,
    __local unsigned int* hit,
    T threshold,
    double actual_selectivity,
    __global unsigned char* const hitmap
  ) {
  unsigned char result = 0;
  T local_karma = karma[get_global_id(0)];
  hit[get_local_id(0)] = (local_karma < threshold) << (get_local_id(0) % 8);

  __local T n[D];
  __local T d[D];
  if(actual_selectivity == 0.0){
    
    //Calculate the ingredients for the formula
    if(get_local_id(0) < D){
      T factor1 = (query[get_local_id(0)*2+1]-query[get_local_id(0)*2])/(bandwidth[get_local_id(0)]*M_SQRT2);
      n[get_local_id(0)] = erf(factor1);
      d[get_local_id(0)] = erf(factor1/2);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    T pmax = 1;
    T max_frac = 0;

    for(int i = 0; i < D; i++){
      pmax *= n[i];
      max_frac = max(max_frac,n[i]/d[i]);
    }
    pmax *= max_frac;
    
    T local_contribution = local_results[get_global_id(0)]; 
    hit[get_local_id(0)] |= (local_contribution > pmax) << (get_local_id(0) % 8);
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if(get_local_id(0) % 8 < 4){
    hit[get_local_id(0)] |= hit[get_local_id(0)+4];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  
  if(get_local_id(0) % 8 < 2){
    hit[get_local_id(0)] |= hit[get_local_id(0)+2];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  
  if(get_local_id(0) % 8 < 1){
    hitmap[get_group_id(0)*get_local_size(0)/8 + get_local_id(0) / 8] = (hit[get_local_id(0)] | hit[get_local_id(0)+1]) & 0xFF;
  }
}
