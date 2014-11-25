#pragma OPENCL EXTENSION cl_khr_fp64: enable

#ifndef TYPE_DEFINED_
  #if (TYPE == 4)
    typedef float T;
  #elif (TYPE == 8)
    typedef double T;
  #endif
  #define TYPE_DEFINED_
#endif /* TYPE_DEFINED */

__kernel void sum_seq(
   __global const T* const data,
   const unsigned int data_offset,
   const unsigned int elements,
   __global T* const result,
   const unsigned int result_offset
){
   T agg = 0;
   for (unsigned i=0; i<elements; ++i) {
     agg += data[data_offset + i];
   }
   result[result_offset] = agg;
}

__kernel void sum_par(
   __global const T* data,
   __local T* scratch,
   __global T* result,
   unsigned int values_per_thread,
   unsigned int nr_of_values
){
   unsigned int local_id = get_local_id(0);
   unsigned int global_id = get_global_id(0);
   // Each thread first does a sequential aggregation within a register.
   T agg = 0;
   #ifdef DEVICE_GPU
      // On the GPU we use a strided access pattern, so that the GPU
      // can coalesce memory access.
      unsigned int group_start = get_local_size(0)*values_per_thread*get_group_id(0);
      for (unsigned int i=0; i < values_per_thread; ++i) {
        unsigned int pos = group_start + i*get_local_size(0) + local_id;
        agg += pos < nr_of_values ? data[pos] : 0;
      }
   #elif defined DEVICE_CPU
      // On the CPU we use a sequential access pattern to keep cache misses
      // local per thread.
      for (unsigned int i=0; i < values_per_thread; ++i) {
        unsigned int pos = values_per_thread * global_id + i;
        if (pos < nr_of_values) {
          agg += data[pos];
        }
      }
   #endif

   // Push the local result to the local buffer, so we can aggregate recursively.
   scratch[local_id] = agg;
   barrier(CLK_LOCAL_MEM_FENCE);

   // Recursively aggregate the tuples in memory.
   if (get_local_size(0) >= 4096) {
      if (local_id < 2048) scratch[local_id] += scratch[local_id + 2048];
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 2048) {
      if (local_id < 1024) scratch[local_id] += scratch[local_id + 1024];
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 1024) {
      if (local_id < 512) scratch[local_id] += scratch[local_id + 512];
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 512) {
      if (local_id < 256) scratch[local_id] += scratch[local_id + 256];
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 256) {
     if (local_id < 128) scratch[local_id] += scratch[local_id + 128];
     barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 128) {
      if (local_id < 64) scratch[local_id] += scratch[local_id + 64];
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (local_id < 32) scratch[local_id] += scratch[local_id + 32];
   barrier(CLK_LOCAL_MEM_FENCE);

   if (local_id < 16) scratch[local_id] += scratch[local_id + 16];
   barrier(CLK_LOCAL_MEM_FENCE);

   if (local_id < 8) scratch[local_id] += scratch[local_id + 8];
   barrier(CLK_LOCAL_MEM_FENCE);

   if (local_id < 4) scratch[local_id] += scratch[local_id + 4];
   barrier(CLK_LOCAL_MEM_FENCE);

   if (local_id < 2) scratch[local_id] += scratch[local_id + 2];
   barrier(CLK_LOCAL_MEM_FENCE);

   // Ok, we are done, write the result back.
   if (local_id == 0) result[get_group_id(0)] = scratch[0] + scratch[1];
}
