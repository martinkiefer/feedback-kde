#pragma OPENCL EXTENSION cl_khr_fp64: enable

#ifndef TYPE_DEFINED_
  #if (TYPE == 4)
    typedef float T;
  #elif (TYPE == 8)
    typedef double T;
  #endif
  #define TYPE_DEFINED_
#endif /* TYPE_DEFINED */

__kernel void applyGradient(
	__global T* bandwidth,
	__global const T* gradient,
	T factor
	) {
	T tmp = bandwidth[get_global_id(0)];
	tmp -= factor * gradient[get_global_id(0)];
	// Ensure we never move into negative bandwidths.
	bandwidth[get_global_id(0)] = max(tmp, (T)0.0001);
}

// KERNELS FOR BATCH GRADIENT COMPUTATION

#ifndef LOG_BANDWIDTH                                                                                                    
  #define BATCH_GRADIENT_COMMON()                                             \
    if (get_global_id(0) >= nr_of_observations) return;                       \
    /* Initialize the scratch spaces. */                                      \
    for (unsigned int i=0; i<D; ++i) {                                        \
      lower_bound_scratch[D * get_local_id(0) + i] =                          \
	  ranges[2 * (D * get_global_id(0) + i)];                             \
      upper_bound_scratch[D*get_local_id(0) + i] =                            \
	  ranges[2 * (D * get_global_id(0) + i) + 1];                         \
      gradient_scratch[D * get_local_id(0) + i] = 0;                          \
    }                                                                         \
    T estimate = 0;                                                           \
    /* Iterate over all sample points. */                                     \
    for (unsigned int i=0; i<nr_of_data_points; ++i) {                        \
      /* Compute the local contributions from this data point. */             \
      T local_contribution = 1.0;                                             \
      T local_gradient[D];                                                    \
      for (unsigned int j=0; j<D; ++j) {                                      \
	local_gradient[j] = 1.0;                                              \
      }                                                                       \
      for (unsigned int j=0; j<D; ++j) {                                      \
	T val = data[D*i + j];                                                \
	T m = mean[j];                                                        \
	T v = variance[j];                                                    \
	T h = bandwidth[j];                                                   \
	T lo = (lower_bound_scratch[D*get_local_id(0) + j]-m)/v - val;        \
	T hi = (upper_bound_scratch[D*get_local_id(0) + j]-m)/v - val;        \
	T factor1  = isinf(lo) ? 0 : lo * exp((T)-1.0 * lo * lo / (2*h*h));   \
	  factor1 -= isinf(hi) ? 0 : hi * exp((T)-1.0 * hi * hi / (2*h*h));   \
	T factor2 = erf(hi / (M_SQRT2 * h)) - erf(lo / (M_SQRT2 * h));        \
	local_contribution *= factor2;                                        \
	for (unsigned int k=0; k<D; ++k) {                                    \
	  local_gradient[k] *= (k==j) ? factor1 : factor2;                    \
	}                                                                     \
      }                                                                       \
      estimate += local_contribution;                                         \
      for (unsigned int j=0; j<D; ++j) {                                      \
	gradient_scratch[D * get_local_id(0) + j] += local_gradient[j];       \
      }                                                                       \
    }                                                                         \
    estimate /= pow((T)2.0, D) * nr_of_data_points;                           \
    T expected = observations[get_global_id(0)];                              
#else
  #define BATCH_GRADIENT_COMMON()                                             \
    if (get_global_id(0) >= nr_of_observations) return;                       \
    /* Initialize the scratch spaces. */                                      \
    for (unsigned int i=0; i<D; ++i) {                                        \
      lower_bound_scratch[D * get_local_id(0) + i] =                          \
	  ranges[2 * (D * get_global_id(0) + i)];                             \
      upper_bound_scratch[D*get_local_id(0) + i] =                            \
	  ranges[2 * (D * get_global_id(0) + i) + 1];                         \
      gradient_scratch[D * get_local_id(0) + i] = 0;                          \
    }                                                                         \
    T estimate = 0;                                                           \
    /* Iterate over all sample points. */                                     \
    for (unsigned int i=0; i<nr_of_data_points; ++i) {                        \
      /* Compute the local contributions from this data point. */             \
      T local_contribution = 1.0;                                             \
      T local_gradient[D];                                                    \
      for (unsigned int j=0; j<D; ++j) {                                      \
	local_gradient[j] = 1.0;                                              \
      }                                                                       \
      for (unsigned int j=0; j<D; ++j) {                                      \
	T val = data[D*i + j];                                                \
	T m = mean[j];                                                        \
	T v = variance[j];                                                    \
	T h = bandwidth[j];                                                   \
	T lo = (lower_bound_scratch[D*get_local_id(0) + j]-m)/v - val;        \
	T hi = (upper_bound_scratch[D*get_local_id(0) + j]-m)/v - val;        \
	T factor1  = isinf(lo) ? 0 : lo * exp((T)-1.0 * lo * lo / (2*exp(h)*exp(h)));   \
	  factor1 -= isinf(hi) ? 0 : hi * exp((T)-1.0 * hi * hi / (2*exp(h)*exp(h)));   \
	T factor2 = erf(hi / (M_SQRT2 * exp(h))) - erf(lo / (M_SQRT2 * exp(h)));        \
	local_contribution *= factor2;                                        \
	for (unsigned int k=0; k<D; ++k) {                                    \
	  local_gradient[k] *= (k==j) ? factor1 : factor2;                    \
	}                                                                     \
      }                                                                       \
      estimate += local_contribution;                                         \
      for (unsigned int j=0; j<D; ++j) {                                      \
	gradient_scratch[D * get_local_id(0) + j] += local_gradient[j];       \
      }                                                                       \
    }                                                                         \
    estimate /= pow((T)2.0, D) * nr_of_data_points;                           \
    T expected = observations[get_global_id(0)];                                                                                                              
#endif
    
__kernel void computeBatchGradientAbsolute(
    __global T* data,
    unsigned int nr_of_data_points,
    __global T* ranges,
    __global T* observations,
    unsigned int nr_of_observations,
    __global T* bandwidth,
    // Scratch space.
    __local T* lower_bound_scratch,
    __local T* upper_bound_scratch,
    __local T* gradient_scratch,
    // Result.
    __global T* cost_values,
    __global T* gradient,
    unsigned int gradient_stride,
    unsigned int nrows,  /* Number of rows in table */
    __global const T* const mean,
    __global const T* const variance
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  T error = estimate - expected;
  T factor = error == 0 ? 0 : (error < 0 ? -1.0 : 1.0);
  cost_values[get_global_id(0)] = error * factor;
  // Finally, write the gradient from this observation to global memory.
  for (unsigned int i=0; i<D; ++i) {
    gradient[i * gradient_stride + get_global_id(0)] =
        gradient_scratch[get_local_id(0) * D + i] * factor;
  }
}

__kernel void computeBatchGradientRelative(
    __global T* data,
    unsigned int nr_of_data_points,
    __global T* ranges,
    __global T* observations,
    unsigned int nr_of_observations,
    __global T* bandwidth,
    // Scratch space.
    __local T* lower_bound_scratch,
    __local T* upper_bound_scratch,
    __local T* gradient_scratch,
    // Result.
    __global T* cost_values,
    __global T* gradient,
    unsigned int gradient_stride,
    unsigned int nrows,  /* Number of rows in table */
    __global const T* const mean,
    __global const T* const variance
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  T error = estimate - expected;
  T factor = (error < 0 ? -1.0 : 1.0) / (1e-10 + expected);
  cost_values[get_global_id(0)] = error * factor;
  // Finally, write the gradient from this observation to global memory.
  for (unsigned int i=0; i<D; ++i) {
    gradient[i * gradient_stride + get_global_id(0)] =
        gradient_scratch[get_local_id(0) * D + i] * factor;
  }
}

__kernel void computeBatchGradientSquaredRelative(
    __global T* data,
    unsigned int nr_of_data_points,
    __global T* ranges,
    __global T* observations,
    unsigned int nr_of_observations,
    __global T* bandwidth,
    // Scratch space.
    __local T* lower_bound_scratch,
    __local T* upper_bound_scratch,
    __local T* gradient_scratch,
    // Result.
    __global T* cost_values,
    __global T* gradient,
    unsigned int gradient_stride,
    unsigned int nrows,  /* Number of rows in table */
    __global const T* const mean,
    __global const T* const variance
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  T error = (estimate - expected) / (1e-10 + expected);
  T factor = 2 * error / (1e-10 + expected);
  cost_values[get_global_id(0)] = error * error;
  // Finally, write the gradient from this observation to global memory.
  for (unsigned int i=0; i<D; ++i) {
    gradient[i * gradient_stride + get_global_id(0)] =
        gradient_scratch[get_local_id(0) * D + i] * factor;
  }
}

__kernel void computeBatchGradientQuadratic(
    __global T* data,
    unsigned int nr_of_data_points,
    __global T* ranges,
    __global T* observations,
    unsigned int nr_of_observations,
    __global T* bandwidth,
    // Scratch space.
    __local T* lower_bound_scratch,
    __local T* upper_bound_scratch,
    __local T* gradient_scratch,
    // Result.
    __global T* cost_values,
    __global T* gradient,
    unsigned int gradient_stride,
    unsigned int nrows,  /* Number of rows in table */
    __global const T* const mean,
    __global const T* const variance
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  T error = estimate - expected;
  T factor = 2 * error;
  cost_values[get_global_id(0)] = error * error;
  // Finally, write the gradient from this observation to global memory.
  for (unsigned int i=0; i<D; ++i) {
    gradient[i * gradient_stride + get_global_id(0)] =
        factor * gradient_scratch[get_local_id(0) * D + i];
  }
}

__kernel void computeBatchGradientQ(
    __global T* data,
    unsigned int nr_of_data_points,
    __global T* ranges,
    __global T* observations,
    unsigned int nr_of_observations,
    __global T* bandwidth,
    // Scratch space.
    __local T* lower_bound_scratch,
    __local T* upper_bound_scratch,
    __local T* gradient_scratch,
    // Result.
    __global T* cost_values,
    __global T* gradient,
    unsigned int gradient_stride,
    unsigned int nrows,  /* Number of rows in table */
    __global const T* const mean,
    __global const T* const variance
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  T error = log(1 + estimate) - log(1 + expected);
  T factor = 2 * error / (1 + estimate);
  cost_values[get_global_id(0)] = error * error;
  // Finally, write the gradient from this observation to global memory.
  for (unsigned int i=0; i<D; ++i) {
    gradient[i * gradient_stride + get_global_id(0)] =
        factor * gradient_scratch[get_local_id(0) * D + i];
  }
}
