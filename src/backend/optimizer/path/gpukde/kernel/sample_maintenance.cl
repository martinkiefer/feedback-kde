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
