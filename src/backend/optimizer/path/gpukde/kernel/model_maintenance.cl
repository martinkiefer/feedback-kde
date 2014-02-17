#ifndef TYPE_DEFINED_
  #if (TYPE == float)
    typedef float T;
  #elif (TYPE == double)
    tpyedef double T;
  #endif
  #define TYPE_DEFINED_
#endif /* TYPE_DEFINED */

// KERNELS FOR ONLINE GRADIENT COMPUTATION

__kernel void computeSingleGradient(
  __global const T* const data,
  unsigned int items_in_sample,
  __global const T* const range,
  __global const T* const bandwidth,
  __local T* scratch,
  __global T* gradient,
  unsigned int gradient_stride
  ) {

  if (get_global_id(0) >= items_in_sample) return;
  // First compute the factors.
  T res = 1.0f;
  for (unsigned int i=0; i<D; ++i) {
    T val = data[D*get_global_id(0) + i];
    T h = bandwidth[i];
    h = h <= 0 ? 0.0001f : h;
    T lo = range[2*i] - val;
    T up = range[2*i + 1] - val;

    T factor1 = isinf(lo) ? 0 : (lo / (sqrt(2 * M_PI) * pow(h, 1.5f))
                * exp(-1.0f * lo * lo / (2*h)));
    factor1 -= isinf(up) ? 0 : (up / (sqrt(2 * M_PI) * pow(h, 1.5f))
               * exp(-1.0f * up * up / (2*h)));
    T factor2 = erf(up / sqrt(2*h)) - erf(lo / sqrt(2*h));

    res *= factor2;
    scratch[D*get_local_id(0) + i]  =  factor2 == 0 ? 0 : factor1 / factor2;
  }
  // Now compute the gradient.
  for (unsigned int i=0; i<D; ++i) {
    T grad = res;
    grad *= scratch[D*get_local_id(0) + i];
    gradient[i * gradient_stride + get_global_id(0)] = res;
  }
}

__kernel void accumulateGradient(
    __global T* gradient_accumulator,
    __global const T* gradient_contribution,
    T factor) {
  gradient_accumulator[get_global_id(0)] += factor * gradient_contribution[get_global_id(0)];
}

__kernel void applyGradient(
	__global T* bandwidth,
	__global const T* gradient,
	T factor
	) {
	T tmp = bandwidth[get_global_id(0)];
	tmp -= factor * gradient[get_global_id(0)];
	// Ensure we never move into negative bandwidths.
	bandwidth[get_global_id(0)] = max(tmp, 0.0001f);
}

// KERNELS FOR BATCH GRADIENT COMPUTATION

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
  T estimation = 0;                                                         \
  T parameter_scratch[D];                                                   \
  /* Iterate over all sample points. */                                     \
  for (unsigned int i=0; i<nr_of_data_points; ++i) {                        \
    /* Compute the local contributions from this data point. */             \
    T local_contribution = 1.0f;                                            \
    for (unsigned int j=0; j<D; ++j) {                                      \
      T val = data[D*i + j];                                                \
      T h = bandwidth[j] <= 0 ? 0.0001f : bandwidth[j];                     \
      T lo = lower_bound_scratch[D*get_local_id(0) + j] - val;              \
      T up = upper_bound_scratch[D*get_local_id(0) + j] - val;              \
      T factor1 = isinf(lo) ? 0 : (lo / (sqrt(2 * M_PI) * pow(h, 1.5f))     \
                  * exp(-1.0f * lo * lo / (2*h)));                          \
      factor1 -= isinf(up) ? 0 : (up / (sqrt(2 * M_PI) * pow(h, 1.5f))      \
                 * exp(-1.0f * up * up / (2*h)));                           \
      T factor2 = erf(up / sqrt(2*h)) - erf(lo / sqrt(2*h));                \
      local_contribution *= factor2;                                        \
      parameter_scratch[j] = factor2 == 0 ? 0 : factor1 / factor2;          \
    }                                                                       \
    estimation += local_contribution;                                       \
    for (unsigned int j=0; j<D; ++j) {                                      \
      gradient_scratch[get_local_id(0) * D + j] +=                          \
          local_contribution * parameter_scratch[j];                        \
    }                                                                       \
  }                                                                         \
  estimation /= pow(2.0f, D) * nr_of_data_points;                           \


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
    unsigned int gradient_stride
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  T error = estimation - observations[get_global_id(0)];
  T factor = error < 0 ? -1.0f : 1.0f;
  cost_values[get_global_id(0)] = error * factor;
  factor /= pow(2.0f, D) * nr_of_data_points;
  // Finally, write the gradient from this observation to global memory.
  for (unsigned int i=0; i<D; ++i) {
    gradient[i * gradient_stride + get_global_id(0)] =
        gradient_scratch[get_local_id(0) * D + i];
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
    unsigned int gradient_stride
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  T error = estimation - observations[get_global_id(0)];
  T factor = (error < 0 ? -1.0f : 1.0f) / (0.001f + observations[get_global_id(0)]);
  cost_values[get_global_id(0)] = error * factor;
  factor /= pow(2.0f, D) * nr_of_data_points;
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
    unsigned int gradient_stride
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  T error = estimation - observations[get_global_id(0)];
  T factor = 2 * error;
  cost_values[get_global_id(0)] = error * error;
  factor /= pow(2.0f, D) * nr_of_data_points;
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
    unsigned int gradient_stride
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  T error = log(0.0001f + estimation) - log(0.0001f + observations[get_global_id(0)]);
  T factor = 2 * error / (0.0001f + observations[get_global_id(0)]);
  cost_values[get_global_id(0)] = error * error;
  factor /= pow(2.0f, D) * nr_of_data_points;
  // Finally, write the gradient from this observation to global memory.
  for (unsigned int i=0; i<D; ++i) {
    gradient[i * gradient_stride + get_global_id(0)] =
        factor * gradient_scratch[get_local_id(0) * D + i];
  }
}
