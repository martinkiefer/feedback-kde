#ifndef TYPE_DEFINED_
  #if (TYPE == 4)
    typedef float T;
  #elif (TYPE == 8)
    typedef double T;
  #endif
  #define TYPE_DEFINED_
#endif /* TYPE_DEFINED */


// Computes partial gradient contributions from each sample item.
__kernel void computePartialGradient(
    __global const T* const data,
    unsigned int items_in_sample,
    __global const T* const range,
    __global const T* const bandwidth,
    __global const T* const bandwidth_delta,
    __local T* scratch,
    __global T* gradient,
    unsigned int gradient_stride,
    __global T* result
    ) {
  if (get_global_id(0) >= items_in_sample) return;
  // First compute the factors.
  T res = 1.0f;
  for (unsigned int i=0; i<D; ++i) {
    T val = data[D*get_global_id(0) + i];
    T h = bandwidth[i];
    if (bandwidth_delta) h += bandwidth_delta[i];
    h = h <= 0 ? 1e-5 : h;
    T lo = range[2*i] - val;
    T up = range[2*i + 1] - val;

    T factor1 = isinf(lo) ? 0 : (lo / (sqrt(2 * M_PI) * pow(h, (T)1.5))
                * exp(-1.0f * lo * lo / (2*h)));
    factor1 -= isinf(up) ? 0 : (up / (sqrt(2 * M_PI) * pow(h, (T)1.5))
               * exp(-1.0f * up * up / (2*h)));
    T factor2 = erf(up / sqrt(2*h)) - erf(lo / sqrt(2*h));

    res *= factor2;
    scratch[D*get_local_id(0) + i]  =  factor2 == 0 ? 0 : factor1 / factor2;
  }

  // Compute the gradient.
  for (unsigned int i=0; i<D; ++i) {
    T grad = res;
    grad *= scratch[D*get_local_id(0) + i];
    gradient[i * gradient_stride + get_global_id(0)] = grad;
  }

  // If requested, write out the result as well.
  if (result) result[get_global_id(0)] = res;
}

// This is a very simple kernel that only multiplies a given value by a factor.
__kernel void finalizeEstimate(
    __global T* result,
    T factor
    ) {
  result[0] *= factor;
}

__kernel void accumulateOnlineBuffers(
    __global const T* gradient,
    __global const T* shifted_gradient,
    T gradient_factor,
    T shifted_gradient_factor,
    __global const T* running_gradient_average,
    __global T* gradient_accumulator,
    __global T* squared_gradient_accumulator,
    __global T* hessian_accumulator,
    __global T* squared_hessian_accumulator
    ) {
  unsigned int i = get_global_id(0);
  T grad = gradient_factor * gradient[i];
  T shift_grad = shifted_gradient_factor * shifted_gradient[i];
  // First, compute the hessian approximation via finite differences.
  T dx = running_gradient_average[i];
  T hess = dx == 0 ? 0 : fabs ( (grad - shift_grad) / dx );
  // Now update the accumulators.
  gradient_accumulator[i] += grad;
  squared_gradient_accumulator[i] += grad * grad;
  hessian_accumulator[i] += hess;
  squared_hessian_accumulator[i] += hess * hess;
}

__kernel void initializeOnlineEstimate(
    __global T* gradient_accumulator,
    __global T* squared_gradient_accumulator,
    __global T* hessian_accumulator,
    __global T* squared_hessian_accumulator,
    __global T* running_gradient_average,
    __global T* running_squared_gradient_average,
    __global T* running_hessian_average,
    __global T* running_squared_hessian_average,
    unsigned int mini_batch_size
    ) {
  unsigned int i = get_global_id(0);

  // Fetch and normalize the latest observations.
  T g = gradient_accumulator[i] / mini_batch_size;
  T gs = squared_gradient_accumulator[i] / mini_batch_size;
  T h = hessian_accumulator[i] / mini_batch_size;
  T hs = squared_hessian_accumulator[i] / mini_batch_size;

  // We now use these to initialze our running averages.
  running_gradient_average[i] = g;
  running_squared_gradient_average[i] = max(1e-5, gs);
  running_hessian_average[i] = max(1e-5, h);
  running_squared_hessian_average[i] = max(1e-5, hs);

  // Reset the accumulators.
  gradient_accumulator[i] = 0;
  squared_gradient_accumulator[i] = 0;
  hessian_accumulator[i] = 0;
  squared_hessian_accumulator[i] = 0;
}

__kernel void updateOnlineEstimate(
    __global T* gradient_accumulator,
    __global T* squared_gradient_accumulator,
    __global T* hessian_accumulator,
    __global T* squared_hessian_accumulator,
    __global T* running_gradient_average,
    __global T* running_squared_gradient_average,
    __global T* running_hessian_average,
    __global T* running_squared_hessian_average,
    __global T* current_time_factor,
    __global T* bandwidth,
    unsigned int mini_batch_size
    ) {
  unsigned int i = get_global_id(0);

  // Fetch and normalize the latest observations.
  T g = gradient_accumulator[i] / mini_batch_size;
  T gs = squared_gradient_accumulator[i] / mini_batch_size;
  T h = hessian_accumulator[i] / mini_batch_size;
  T hs = squared_hessian_accumulator[i] / mini_batch_size;

  // Fetch the running averages.
  T g_ = running_gradient_average[i];
  T gs_ = running_squared_gradient_average[i];
  T h_ = running_hessian_average[i];
  T hs_ = running_squared_hessian_average[i];

  // Time factor.
  T t = current_time_factor[i];

  // First, we check whether this is an outlier.
  char is_outlier = fabs(g - g_) > 2 * sqrt(gs_ - g_ * g_);
  is_outlier |= fabs(h - h_) > 2 * sqrt(hs_ - h_ * h_);
  // For outliers, we increase the time window, so they don't impact the
  // estimate very strongly.
  t += is_outlier ? 1 : 0;
  T tinv = 1/t;

  // Update the moving averages.
  g_ = (1 - tinv) * g_ + tinv * g;
  gs_ = (1 - tinv) * gs_ + tinv * gs;
  h_ = (1 - tinv) * h_ + tinv * h;
  hs_ = (1 - tinv) * hs_ + tinv * hs;

  // Compensate for negative values:
  gs_ = max(1e-4, gs_);
  h_ = max(1e-4, h_);
  hs_ = max(1e-4, hs_);

  // Estimate the learning rate.
  T learning_rate = (h_ / hs_) * (mini_batch_size * g_ * g_) / (gs_ + (mini_batch_size - 1) * g_ * g_);

  // Update the time factor.
  t = (1 - g_ * g_ / gs_) * t + 1;

  // Update the bandwidth.
  T b = bandwidth[i] - learning_rate * g;
  b = max(1e-5, b);   // Never allow negative bandwidths.
  bandwidth[i] = b;

  // Write back all running estimates.
  running_gradient_average[i] = g_;
  running_squared_gradient_average[i] = gs_;
  running_hessian_average[i] = h_;
  running_squared_hessian_average[i] = hs_;
  current_time_factor[i] = t;

  // And zero out the accumulators.
  gradient_accumulator[i] = 0;
  squared_gradient_accumulator[i] = 0;
  hessian_accumulator[i] = 0;
  squared_hessian_accumulator[i] = 0;
}
