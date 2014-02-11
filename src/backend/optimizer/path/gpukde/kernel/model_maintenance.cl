// KERNELS FOR ONLINE GRADIENT COMPUTATION

__kernel void computeSingleGradient(
	__global const float* const data,
	unsigned int items_in_sample,
	__global const float* const range,
	__global const float* const bandwidth,
	__local float* scratch,
	__global float* gradient,
	unsigned int gradient_stride
	) {
	
	if (get_global_id(0) >= items_in_sample) return;
	// First compute the factors.
	float res = 1.0f;
	for (unsigned int i=0; i<D; ++i) {
		float val = data[D*get_global_id(0) + i];
		float h = bandwidth[i];
		float lo = range[2*i] - val;
		float up = range[2*i + 1] - val;
		
		float factor1 = lo / (sqrt(2 * M_PI)*pow(h, 1.5f)) * exp(-1.0f * lo * lo / (2*h));
		factor1 -= up / (sqrt(2*M_PI)*pow(h,1.5f)) * exp(-1.0f * up * up / (2*h));	
		float factor2 = erf(up / sqrt(2*h)) - erf(lo / sqrt(2*h));
		
		res *= factor2;
		scratch[D*get_local_id(0) + i]  = factor1 / factor2;
	}
	// Now compute the gradient.
	for (unsigned int i=0; i<D; ++i) {
		float grad = res;
		grad *= scratch[D*get_local_id(0) + i];
		gradient[i * gradient_stride + get_global_id(0)] = res;
	}
}

__kernel void applyGradient(
	__global float* bandwidth,
	__global const float* gradient,
	char cap_to_positive,
	float factor
	) {
	float tmp = bandwidth[get_global_id(0)];
	tmp += factor * gradient[get_global_id(0)];
	// If cap_to_positive is set, we do not allow non-positive values.
	tmp = cap_to_positive ? max(tmp, 0.00001f) : tmp;
	bandwidth[get_global_id(0)] = tmp;
}

// KERNELS FOR BATCH GRADIENT COMPUTATION

#define BATCH_GRADIENT_COMMON()                                           \
  if (get_global_id(0) > nr_of_observations) return;                      \
  /* Initialize the scratch spaces. */                                    \
  for (unsigned int i=0; i<D; ++i) {                                      \
    lower_bound_scratch[D * get_local_id(0) + i] =                        \
        ranges[2 * get_global_id(0)];                                     \
    upper_bound_scratch[D*get_local_id(0) + i] =                          \
        ranges[2 * get_global_id(0) + 1];                                 \
    gradient_scratch[D*get_local_id(0) + i] = 0;                          \
  }                                                                       \
  float estimation = 0;                                                   \
  float parameter_scratch[D];                                             \
  /* Iterate over all sample points. */                                   \
  for (unsigned int i=0; i<nr_of_data_points; ++i) {                      \
    /* Compute the local contributions from this data point. */           \
    float local_contribution = 1.0f;                                      \
    for (unsigned int j=0; j<D; ++j) {                                    \
      float val = data[D*i + j];                                          \
      float h = bandwidth[i];                                             \
      float lo = lower_bound_scratch[D*get_local_id(0) + j] - val;        \
      float up = upper_bound_scratch[D*get_local_id(0) + j] - val;        \
      float factor1 = lo / (sqrt(2 * M_PI) * pow(h, 1.5f))                \
                      * exp(-1.0f * lo * lo / (2*h));                     \
      factor1 -= up / (sqrt(2 * M_PI) * pow(h, 1.5f))                     \
                 * exp(-1.0f * up * up / (2*h));                          \
      float factor2 = erf(up / sqrt(2*h)) - erf(lo / sqrt(2*h));          \
      local_contribution *= factor2;                                      \
      parameter_scratch[D*get_local_id(0) + j] = factor1 / factor2;       \
    }                                                                     \
    estimation += local_contribution;                                     \
    for (unsigned int j=0; j<D; ++j) {                                    \
      gradient_scratch[get_local_id(0) * D + j] +=                        \
          local_contribution * parameter_scratch[D*get_local_id(0) + j];  \
    }                                                                     \
  }                                                                       \
  estimation /= pow(2.0f, D) * nr_of_data_points;                         \


__kernel void computeBatchGradientAbsolute(
    __global float* data,
    unsigned int nr_of_data_points,
    __global float* ranges,
    __global float* observations,
    unsigned int nr_of_observations,
    __global float* bandwidth,
    // Scratch space.
    __local float* lower_bound_scratch,
    __local float* upper_bound_scratch,
    __local float* gradient_scratch,
    // Result.
    __global float* cost_values,
    __global float* gradient
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  float error = estimation - observations[get_global_id(0)];
  float factor = error < 0 ? -1.0f : 1.0f;
  cost_values[get_global_id(0)] = error * factor;
  factor /= pow(2.0f, D) * nr_of_data_points;
  // Finally, write the gradient from this observation to global memory.
  for (unsigned int i=0; i<D; ++i) {
    gradient[D * nr_of_observations + i] =
        factor * gradient_scratch[get_local_id(0) * D + i];
  }
}

__kernel void computeBatchGradientRelative(
    __global float* data,
    unsigned int nr_of_data_points,
    __global float* ranges,
    __global float* observations,
    unsigned int nr_of_observations,
    __global float* bandwidth,
    // Scratch space.
    __local float* lower_bound_scratch,
    __local float* upper_bound_scratch,
    __local float* gradient_scratch,
    // Result.
    __global float* cost_values,
    __global float* gradient
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  float error = estimation - observations[get_global_id(0)];
  float factor = (error < 0 ? -1.0f : 1.0f) / (0.0001f + observations[get_global_id(0)]);
  cost_values[get_global_id(0)] = error * factor;
  factor /= pow(2.0f, D) * nr_of_data_points;
  // Finally, write the gradient from this observation to global memory.
  for (unsigned int i=0; i<D; ++i) {
    gradient[D * nr_of_observations + i] =
        factor * gradient_scratch[get_local_id(0) * D + i];
  }
}

__kernel void computeBatchGradientQuadratic(
    __global float* data,
    unsigned int nr_of_data_points,
    __global float* ranges,
    __global float* observations,
    unsigned int nr_of_observations,
    __global float* bandwidth,
    // Scratch space.
    __local float* lower_bound_scratch,
    __local float* upper_bound_scratch,
    __local float* gradient_scratch,
    // Result.
    __global float* cost_values,
    __global float* gradient
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  float error = estimation - observations[get_global_id(0)];
  float factor = 2 * error;
  cost_values[get_global_id(0)] = error * error;
  factor /= pow(2.0f, D) * nr_of_data_points;
  // Finally, write the gradient from this observation to global memory.
  for (unsigned int i=0; i<D; ++i) {
    gradient[D * nr_of_observations + i] =
        factor * gradient_scratch[get_local_id(0) * D + i];
  }
}

__kernel void computeBatchGradientQ(
    __global float* data,
    unsigned int nr_of_data_points,
    __global float* ranges,
    __global float* observations,
    unsigned int nr_of_observations,
    __global float* bandwidth,
    // Scratch space.
    __local float* lower_bound_scratch,
    __local float* upper_bound_scratch,
    __local float* gradient_scratch,
    // Result.
    __global float* cost_values,
    __global float* gradient
  ) {
  // First, we compute the error-independent parts of the gradient.
  BATCH_GRADIENT_COMMON();
  // Next, compute the estimation error and the gradient scale factor.
  float error = log(0.0001f + estimation) - log(0.0001f + observations[get_global_id(0)]);
  float factor = 2 * error / (0.0001f + observations[get_global_id(0)]);
  cost_values[get_global_id(0)] = error * error;
  factor /= pow(2.0f, D) * nr_of_data_points;
  // Finally, write the gradient from this observation to global memory.
  for (unsigned int i=0; i<D; ++i) {
    gradient[D * nr_of_observations + i] =
        factor * gradient_scratch[get_local_id(0) * D + i];
  }
}
