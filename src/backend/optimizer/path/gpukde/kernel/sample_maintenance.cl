#pragma OPENCL EXTENSION cl_khr_fp64: enable

#ifndef TYPE_DEFINED_
  #if (TYPE == 4)
    typedef float T;
  #elif (TYPE == 8)
    typedef double T;
  #endif
  #define TYPE_DEFINED_
#endif /* TYPE_DEFINED */

// 
__kernel void update_sample_quality_metrics(
    __global const T* local_results,
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
  double improvement = fabs(actual_selectivity - estimated_selectivity);
  improvement -= fabs(actual_selectivity - (T)adjusted_estimate);
  
  // Now compute the karma by normalizing the improvement to [-1,1]
  double local_karma = improvement * sample_size;

  // Now update the array
  karma[get_global_id(0)] *= karma_decay;
  karma[get_global_id(0)] += (1 - karma_decay) * local_karma;
}
