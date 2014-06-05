#ifndef TYPE_DEFINED_
  #if (TYPE == 4)
    typedef float T;
  #elif (TYPE == 8)
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    typedef double T;
  #endif
  #define TYPE_DEFINED_
#endif /* TYPE_DEFINED */

// Applies a single step of stochastic gradient descent to a linear model
// between sample contributions and target selectivity.
__kernel void update_sample_quality_metrics(
	__global const T* local_results,
	__global T* slopes,
	__global T* intercepts,
	T target_result,
	T learning_rate) {
	// Fetch the local result of this data item.
	unsigned int idx = get_global_id(0);
	T local_result = local_results[idx];
	
	// Compute the gradient for the linear model.
	T err = slopes[idx] * local_result + intercepts[idx] - target_result;
	T dSlope = 2*err*local_result;
	T dIntercept = 2*err;
	
	// And apply it to the model parameters.
	slopes[idx] -= learning_rate * dSlope;
	intercepts[idx] -= learning_rate * dIntercept;
}

// 
__kernel void udate_sample_penalties_absolute(
	__global const T* local_results,
	__global T* penalties,
	T sample_size,
	T normalization_factor,
	T estimated_selectivity,
	T actual_selectivity,
	T exponential_forget
	 ) {
	//Get the denormalized selectvity
	T denormalized = estimated_selectivity * sample_size / normalization_factor;
	//Now remove the local contribution of the point
	T without_contribution = denormalized - local_results[get_global_id(0)];
	//Normalize again
	T normalized = without_contribution * normalization_factor / (sample_size - 1); 
	
	//Calculate the error and add to penalty
	penalties[get_global_id(0)] = penalties[get_global_id(0)] * exponential_forget 
					+ fabs(normalized - actual_selectivity) - fabs(estimated_selectivity - actual_selectivity);
}
