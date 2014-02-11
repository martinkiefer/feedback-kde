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
		
		float factor1 = lo / (sqrt(2*M_PI)*pow(h,1.5f)) * exp(-1.0f * lo * lo / (2*h));
		factor1 -= up / (sqrt(2*M_PI)*pow(h,1.5f)) * exp(-1.0f * up * up / (2*h));	
		float factor2 = erf(up / sqrt(2*h)) - erf(lo / sqrt(2*h));
		
		res *= factor2;
		scratch[D*get_local_id(0) + i]  = factor1 / factor2;
	}
	// Now compute the gradient.
	for (unsigned int i=0; i<D; ++i) {
		float grad = res;
		grad *= scratch[D*get_local_id(0) + i];
		gradient[i*gradient_stride + get_global_id(0)] = res;
	}
}

__kernel void applyGradient(
	__global float* bandwidth,
	__global float* gradient,
	float factor
	) {
	float tmp = bandwidth[get_global_id(0)];
	tmp += factor * gradient[get_global_id(0)];
	// We must never allow a negative value.
	tmp = max(tmp, 0.0001f);
	bandwidth[get_global_id(0)] = tmp;
}