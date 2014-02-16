#ifndef TYPE_DEFINED_
  #if (TYPE == float)
    typedef float T;
  #elif (TYPE == double)
    tpyedef double T;
  #endif
  #define TYPE_DEFINED_
#endif /* TYPE_DEFINED */

__kernel void init_zero(
	__global T* data) {
	data[get_global_id(0)] = 0;
}

__kernel void init_one(
	__global T* data) {
	data[get_global_id(0)] = 1.0f;
}
