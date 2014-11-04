#ifndef TYPE_DEFINED_
  #if (TYPE == 4)
    typedef float T;
  #elif (TYPE == 8)
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    typedef double T;
  #endif
  #define TYPE_DEFINED_
#endif /* TYPE_DEFINED */

/* Suggestions from Geoffrey Hinton's slides for mini-batch gradient descent. */
#define STEP_MAX_LIMIT 50
#define STEP_MIN_LIMIT 10^-6
#define STEP_DECREASE 0.5
#define STEP_INCREASE 1.2
    
// Computes partial gradient contributions from each sample item.
__kernel void computePartialGradient(
    __global const T* const data,
    unsigned int items_in_sample,
    __global const T* const range,
    __global const T* const bandwidth,
    __global const T* const running_gradient_average,
    __local T* scratch,
    __global T* gradient,
    unsigned int gradient_stride,
    __global T* estimate
    ) {
  if (get_global_id(0) >= items_in_sample) return;
  // First compute the factors.
  T local_result = 1.0;
  for (unsigned int i=0; i<D; ++i) {
    scratch[D*get_local_id(0) + i] = 1.0;
  }
  for (unsigned int i=0; i<D; ++i) {
    T val = data[D*get_global_id(0) + i];
    T h = bandwidth[i];
    if (running_gradient_average) h += running_gradient_average[i];
    h = h <= 0 ? 1e-10 : h; // Cap H to positive values.
    T lo = range[2*i] - val;
    T hi = range[2*i + 1] - val;

    T factor1 = isinf(lo) ? 0 : lo * exp((T)-1.0 * lo * lo / (2*h*h));
    factor1  -= isinf(hi) ? 0 : hi * exp((T)-1.0 * hi * hi / (2*h*h));
    T factor2 = erf(hi / (M_SQRT2 * h)) - erf(lo / (M_SQRT2 * h));
    local_result *= factor2;
    for (unsigned int j=0; j<D; ++j) {
      scratch[D*get_local_id(0) + j] *= (j==i) ? factor1 : factor2;
    }
  }

  // Compute the gradient.
  for (unsigned int i=0; i<D; ++i) {
    gradient[i * gradient_stride + get_global_id(0)] =
        scratch[D*get_local_id(0) + i];
  }

  // If requested, write out the result estimate as well.
  if (estimate) estimate[get_global_id(0)] = local_result;
}

// This is a very simple kernel that only multiplies a given value by a factor.
__kernel void finalizeEstimate(
    __global T* result,
    T factor
    ) {
  result[0] *= factor;
}

__kernel void accumulateVsgdOnlineBuffers(
    __global const T* gradient,
    __global const T* shifted_gradient,
    T gradient_factor,
    T shifted_gradient_factor,
    __global const T* bandwidth,
    __global const T* running_gradient_average,
    __global T* gradient_accumulator,
    __global T* squared_gradient_accumulator,
    __global T* hessian_accumulator,
    __global T* squared_hessian_accumulator
    ) {
  unsigned int i = get_global_id(0);
  // We need the bandwidth and the shifted bandwidth to scale the gradient.
  T h = bandwidth[i];
  T hs = h + running_gradient_average[i];
  h = h <= 0 ? 1e-10 : h;
  hs = hs <= 0 ? 1e-10 : hs;
  T dh = hs - h;
  // Now scale the gradient and the shifted gradient.
  T grad = gradient_factor * gradient[i] / (h * h);
  T shift_grad = shifted_gradient_factor * shifted_gradient[i] / (hs * hs);
  // First, compute the hessian approximation via finite differences.
  T hess = dh == 0 ? 0 : fabs ( (grad - shift_grad) / dh );
  // Now update the accumulators.
  gradient_accumulator[i] += grad;
  squared_gradient_accumulator[i] += grad * grad;
  hessian_accumulator[i] += hess;
  squared_hessian_accumulator[i] += hess * hess;
}

__kernel void accumulateRmspropOnlineBuffers(
    __global const T* gradient,
    T gradient_factor,
    __global const T* bandwidth,
    __global T* gradient_accumulator
    ) {
  unsigned int i = get_global_id(0);
  // For rmsprop the bandwidth
  T h = bandwidth[i];

  // Now scale the gradient and the shifted gradient.
  T grad = gradient_factor * gradient[i] / (h * h);
  gradient_accumulator[i] += grad;
}


__kernel void initializeVsgdOnlineEstimate(
    __global T* gradient_accumulator,
    __global T* squared_gradient_accumulator,
    __global T* hessian_accumulator,
    __global T* squared_hessian_accumulator,
    __global T* running_gradient_average,
    __global T* running_squared_gradient_average,
    __global T* running_hessian_average,
    __global T* running_squared_hessian_average,
    __global T* current_time_factor,
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
  running_squared_gradient_average[i] = max((T)1e-5, gs);
  running_hessian_average[i] = max((T)1e-5, h);
  running_squared_hessian_average[i] = max((T)1e-5, hs);
  
  // Initialize the time factor to two, so we initially keep some information.
  current_time_factor[i] = 2;

  // Reset the accumulators.
  gradient_accumulator[i] = 0;
  squared_gradient_accumulator[i] = 0;
  hessian_accumulator[i] = 0;
  squared_hessian_accumulator[i] = 0;
}

//Initializes all necessary fields to an initial value.
__kernel void initializeRmspropOnlineEstimate(
    __global T* gradient_accumulator,
    __global T* last_gradient,
    __global T* learning_rate,
    __global T* running_squared_gradient_average,
    unsigned int mini_batch_size
    ) {
  unsigned int i = get_global_id(0);

  // Fetch and normalize the latest observations.
  T g = gradient_accumulator[i] / mini_batch_size;
  learning_rate[i] = 1.0;
  // We now use these to initialze our running averages.
  running_squared_gradient_average[i] = g*g;
  last_gradient[i] = g;
}


__kernel void updateRmspropOnlineEstimate(
    __global T* gradient_accumulator,
    __global T* last_gradient,
    __global T* running_squared_gradient_average,
    __global T* learning_rate,
    __global T* bandwidth,
    unsigned int mini_batch_size
    ) {
  unsigned int i = get_global_id(0);


  // Fetch and normalize the latest observations.
  T g = gradient_accumulator[i] / mini_batch_size;
  T lg = last_gradient[i];
  T gs = g*g;

  T gs_avg = running_squared_gradient_average[i];
  gs_avg = 0.9 * gs_avg + 0.1 * gs;
  
  T lr = learning_rate[i];

  //Standard Rprob 	
  if(lg * g > 0.0)
    lr = fmin(lr * STEP_INCREASE, STEP_MAX_LIMIT);
  else if(lg * g < 0.0)
    lr = fmax(lr * STEP_DECREASE, STEP_MIN_LIMIT);

  //Rmsprop scales the gradient with the square root of the running squared gradient average.
  T scaled_g = g/sqrt(gs_avg);
  if(scaled_g > 0){
    //If the gradient is postive, we will run into problems. Limit the learning rate in that case
    lr = fmin(bandwidth[i]/scaled_g*0.5,lr); 	
  } 
  bandwidth[i] = bandwidth[i] - scaled_g*lr;
  
  //Zero out the accumulators and write to memory
  gradient_accumulator[i] = 0;
  last_gradient[i] = g;
  running_squared_gradient_average[i] = gs_avg;
  learning_rate[i] = lr;
}


// Computes a single vSGD-fd step.
__kernel void updateVsgdOnlineEstimate(
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
    unsigned int mini_batch_size,
    T learning_boost_rate
    ) {
  unsigned int i = get_global_id(0);

  // Fetch and normalize the latest observations.
  T g = gradient_accumulator[i] / mini_batch_size;
  T gs = squared_gradient_accumulator[i] / mini_batch_size;
  T h = hessian_accumulator[i] / mini_batch_size;
  T hs = squared_hessian_accumulator[i] / mini_batch_size;

  // Fetch the running averages.
  T g_avg = running_gradient_average[i];
  T gs_avg = running_squared_gradient_average[i];
  T h_avg = running_hessian_average[i];
  T hs_avg = running_squared_hessian_average[i];

  // Fetch the time factor.
  T t = current_time_factor[i];

  // Peform outlier detection.
  char is_outlier = fabs(g - g_avg) > 2 * sqrt(gs_avg - g_avg * g_avg);
  is_outlier |= fabs(h - h_avg) > 2 * sqrt(hs_avg - h_avg * h_avg);
  // For outliers, we increase the time window, so they don't impact the
  // estimate too strongly.
  t += is_outlier ? 1 : 0;
  T tinv = 1/t;

  // Update the moving averages.
  g_avg  = (1 - tinv) * g_avg  + tinv * g;
  gs_avg = (1 - tinv) * gs_avg + tinv * gs;
  h_avg  = (1 - tinv) * h_avg  + tinv * h;
  hs_avg = (1 - tinv) * hs_avg + tinv * hs;

  // Compensate for negative values:
  gs_avg = max((T)1e-4, gs_avg);
  h_avg  = max((T)1e-4, h_avg);
  hs_avg = max((T)1e-4, hs_avg);

  // Estimate the learning rate.
  T learning_rate  = (h_avg / hs_avg) * (mini_batch_size * g_avg * g_avg);
    learning_rate /= (gs_avg + (mini_batch_size - 1) * g_avg * g_avg);

  // Update the time factor.
  t = (1 - g_avg * g_avg / gs_avg) * t + 1;

  // Update the bandwidth.
  T b = bandwidth[i] - learning_boost_rate * learning_rate * g;
  b = max((T)1e-10, b);   // Never allow negative bandwidths.
  bandwidth[i] = b;

  // Write back all running estimates.
  running_gradient_average[i] = g_avg;
  running_squared_gradient_average[i] = gs_avg;
  running_hessian_average[i] = h_avg;
  running_squared_hessian_average[i] = hs_avg;
  current_time_factor[i] = t;

  // And zero out the accumulators.
  gradient_accumulator[i] = 0;
  squared_gradient_accumulator[i] = 0;
  hessian_accumulator[i] = 0;
  squared_hessian_accumulator[i] = 0;
}
