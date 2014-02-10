// Uses the Epanechnikov Kernel.
__kernel void epanechnikov_kde(
	__global const float* const data,
	__global float* const result,
	__global const float* const range,
	__global const float* const bandwidth
) {
	float res = 1.0f;
	for (unsigned int i=0; i<D; ++i) {
		// Fetch all required input data.
		float val = data[D*get_global_id(0) + i];
		float h = bandwidth[i];
		float lo = range[2*i];
		float up = range[2*i + 1];
		// If the support is completely contained in the query, the result is completely contained.
		char is_complete = (lo <= (val-h)) && (up >= (val+h));
		// Adjust the boundaries, so we only integrate over the defined area.
		lo = max(lo, val-h);
		up = min(val+h, up);
		// ... and compute the local contribution from this dimension:
		float local_result = (h*h - val*val)*(up - lo);
		local_result += val * (up*up - lo*lo);
		local_result -= (up*up*up - lo*lo*lo) / 3.0f; 
		local_result /= h*h*h;
		// Apply the boundary cases: 
		res *= is_complete ? (4.0f / 3.0f) : (lo < up)*local_result;
	}
	result[get_global_id(0)] = res;
}

// Uses the Gauss Kernel.
__kernel void gauss_kde(
	__global const float* const data,
	__global float* const result,
	__global const float* const range,
	__global const float* const bandwidth
) {
	float res = 1.0f;
	for (unsigned int i=0; i<D; ++i) {
		// Fetch all required input data.
		float val = data[D*get_global_id(0) + i];
		float h = sqrt(2*bandwidth[i]);
		float lo = range[2*i];
		float up = range[2*i + 1];
		// Now compute the local result.
		float local_result = erf((up - val) / h);
		local_result -= erf((lo - val) / h);
		res *= local_result;
	}
	result[get_global_id(0)] = res;
}
