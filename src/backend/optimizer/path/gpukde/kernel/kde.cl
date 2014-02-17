#ifndef TYPE_DEFINED_
  #if (TYPE == 4)
    typedef float T;
  #elif (TYPE == 8)
    typedef double T;
  #endif
  #define TYPE_DEFINED_
#endif /* TYPE_DEFINED */

// Uses the Epanechnikov Kernel.
__kernel void epanechnikov_kde(
	__global const T* const data,
	__global T* const result,
	__global const T* const range,
	__global const T* const bandwidth
) {
	T res = 1.0;
	for (unsigned int i=0; i<D; ++i) {
		// Fetch all required input data.
		T val = data[D*get_global_id(0) + i];
		T h = bandwidth[i];
		T lo = range[2*i];
		T up = range[2*i + 1];
		// If the support is completely contained in the query, the result is completely contained.
		char is_complete = (lo <= (val-h)) && (up >= (val+h));
		// Adjust the boundaries, so we only integrate over the defined area.
		lo = max(lo, val-h);
		up = min(val+h, up);
		// ... and compute the local contribution from this dimension:
		T local_result = (h*h - val*val)*(up - lo);
		local_result += val * (up*up - lo*lo);
		local_result -= (up*up*up - lo*lo*lo) / 3.0;
		local_result /= h*h*h;
		// Apply the boundary cases: 
		res *= is_complete ? (4.0 / 3.0) : (lo < up)*local_result;
	}
	result[get_global_id(0)] = res;
}

// Uses the Gauss Kernel.
__kernel void gauss_kde(
	__global const T* const data,
	__global T* const result,
	__global const T* const range,
	__global const T* const bandwidth
) {
	T res = 1.0;
	for (unsigned int i=0; i<D; ++i) {
		// Fetch all required input data.
		T val = data[D*get_global_id(0) + i];
		T h = sqrt(2*bandwidth[i]);
		T lo = range[2*i];
		T up = range[2*i + 1];
		// Now compute the local result.
		T local_result = erf((up - val) / h);
		local_result -= erf((lo - val) / h);
		res *= local_result;
	}
	result[get_global_id(0)] = res;
}
