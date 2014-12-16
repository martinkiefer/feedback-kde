#pragma OPENCL EXTENSION cl_khr_fp64: enable

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
		local_result *= (lo < up);
		// Apply the boundary cases: 
		res *= is_complete ? (4.0 / 3.0) : local_result;
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
		T h = bandwidth[i];
		T lo = range[2*i] - val;
		T up = range[2*i + 1] - val;
		// Now compute the local result.
#ifndef LOG_BANDWIDTH
		T local_result = erf(up / (M_SQRT2 * h));
		local_result -= erf(lo / (M_SQRT2 * h));
#else
		T local_result = erf(up / (M_SQRT2 * log(h)));
		local_result -= erf(lo / (M_SQRT2 * log(h)));
#endif
		res *= local_result;
	}
	result[get_global_id(0)] = res;
}

// Used to extract all values for a single dimension from the data sample.
__kernel void extract_dimension(
  __global const T* const data,
  __global T* result,
  unsigned int selected_dimension,
  unsigned int dimensions
) {
  T my_value = data[get_global_id(0)*dimensions + selected_dimension];
  result[get_global_id(0)] = my_value;
}

// Used to compute the local contributions to the variance.
__kernel void precompute_variance(
  __global T* data,
  __global T* average_buffer,
  unsigned int dimension,
  unsigned int points_in_sample
) {
  T average = average_buffer[dimension] / points_in_sample;
  T my_value = data[get_global_id(0)] - average;
  data[get_global_id(0)] = my_value * my_value;
}

__kernel void set_scotts_bandwidth(
  __global T* variance_buffer,
  __global T* bandwidth_buffer,
  unsigned int selected_dimension,
  unsigned int dimensions,
  unsigned int points_in_sample
) {
  T var = variance_buffer[selected_dimension] / (points_in_sample - 1);
  T bandwidth = var * pow(
    4.0 / ((dimensions + 2.0) * points_in_sample), 1.0/(dimensions + 4.0));
#ifndef LOG_BANDWIDTH
  bandwidth_buffer[selected_dimension] = bandwidth;
#else
  bandwidth_buffer[selected_dimension] = exp(bandwidth);
#endif
}
