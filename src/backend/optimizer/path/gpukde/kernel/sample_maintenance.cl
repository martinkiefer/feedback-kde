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
    __global T* contribution,
    unsigned int sample_size,
    T normalization_factor,
    double estimated_selectivity,
    double actual_selectivity,
    double karma_decay,
    double contribution_decay
  ) {
  T local_contribution = local_results[get_global_id(0)];

  // Compute the estimate without the current point.
  double adjusted_estimate = estimated_selectivity * sample_size / normalization_factor;
  adjusted_estimate -= local_contribution;
  adjusted_estimate *= normalization_factor / (sample_size - 1);

  // Compute whether this improved or degraded the estimate.
  double improvement = fabs(actual_selectivity - estimated_selectivity);
  improvement -= fabs(actual_selectivity - (T)adjusted_estimate);
  // Now compute the karma, we use the followng, very simple approach:
  //  The karma is -1 if improvement < 0
  //  The karma is +1 if improvement > 0
  double local_karma;
  if (actual_selectivity == 0 && local_contribution > 0.001) {
    local_karma = -1;
  } else {
    local_karma = actual_selectivity == 0 && fabs(improvement) < 0.0005 ? 0 : improvement < 0 ? -1 : 1;
  }

  // Now update the arrays.
  contribution[get_global_id(0)] *= contribution_decay;
  contribution[get_global_id(0)] += (1 - contribution_decay) * local_contribution * normalization_factor;
  karma[get_global_id(0)] *= karma_decay;
  karma[get_global_id(0)] += (1 - karma_decay) * local_karma;
}
