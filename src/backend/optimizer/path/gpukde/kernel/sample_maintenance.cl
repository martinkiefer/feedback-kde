// Applies a single step of stochastic gradient descent to a linear model
// between sample contributions and target selectivity.
__kernel void update_sample_quality_metrics(
	__global const float* local_results,
	__global float* slopes,
	__global float* intercepts,
	float target_result,
	float learning_rate) {
	// Fetch the local result of this data item.
	unsigned int idx = get_global_id(0);
	float local_result = local_results[idx];
	
	// Compute the gradient for the linear model.
	float err = slopes[idx] * local_result + intercepts[idx] - target_result;
	float dSlope = 2*err*local_result;
	float dIntercept = 2*err;
	
	// And apply it to the model parameters.
	slopes[idx] -= learning_rate * dSlope;
	intercepts[idx] -= learning_rate * dIntercept;
}