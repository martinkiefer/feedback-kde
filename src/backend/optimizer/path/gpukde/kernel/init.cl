__kernel void init_zero(
	__global float* data) {
	data[get_global_id(0)] = 0;
}

__kernel void init_one(
	__global float* data) {
	data[get_global_id(0)] = 1.0f;
}