#ifndef TYPE_DEFINED_
  #if (TYPE == 4)
    typedef float T;
  #elif (TYPE == 8)
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
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
   for (unsigned i=0; i<elements; ++i)
      agg += data[data_offset + i];
   result[result_offset] = agg;
}

__kernel void sum_par(
   __global const T* const data,
   __local T* buffer,
   __global T* const result,
   const unsigned int tuples_per_thread
){
   unsigned int local_id = get_local_id(0);
   unsigned int global_id = get_global_id(0);
   // Each thread first does a sequential aggregation within a register.
   T agg = 0;
   #ifdef DEVICE_GPU
      // On the GPU we use a strided access pattern, so that the GPU
      // can coalesc memory access.
      unsigned int group_start = get_local_size(0)*tuples_per_thread*get_group_id(0);
      for (unsigned int i=0; i < tuples_per_thread; ++i)
         agg += data[group_start + i*get_local_size(0) + local_id];
   #elif defined DEVICE_CPU
      // On the CPU we use a sequential access pattern to keep cache misses
      // local per thread.
      for (unsigned int i=0; i < tuples_per_thread; ++i)
         agg += data[tuples_per_thread*global_id + i];
   #endif

   // Push the local result to the local buffer, so we can aggregate the remainder
   // recursively.
   buffer[local_id] = agg;
   barrier(CLK_LOCAL_MEM_FENCE);

   // Recursively aggregate the tuples in memory.
   if (get_local_size(0) >= 4096) {
      if (local_id < 2048)
         buffer[local_id] += buffer[local_id + 2048];
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 2048) {
      if (local_id < 1024)
         buffer[local_id] += buffer[local_id + 1024];
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 1024) {
      if (local_id < 512)
         buffer[local_id] += buffer[local_id + 512];
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 512) {
      if (local_id < 256)
         buffer[local_id] += buffer[local_id + 256];
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 256) {
     if (local_id < 128)
         buffer[local_id] += buffer[local_id + 128];
     barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 128) {
      if (local_id < 64)
         buffer[local_id] += buffer[local_id + 64];
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (local_id < 32)
      buffer[local_id] += buffer[local_id + 32];
   barrier(CLK_LOCAL_MEM_FENCE);

   if (local_id < 16)
      buffer[local_id] += buffer[local_id + 16];
   barrier(CLK_LOCAL_MEM_FENCE);

   if (local_id < 8)
      buffer[local_id] += buffer[local_id + 8];
   barrier(CLK_LOCAL_MEM_FENCE);

   if (local_id < 4)
      buffer[local_id] += buffer[local_id + 4];
   barrier(CLK_LOCAL_MEM_FENCE);

   if (local_id < 2)
      buffer[local_id] += buffer[local_id + 2];
   barrier(CLK_LOCAL_MEM_FENCE);

   // Ok, we are done, write the result back.
   if (local_id == 0) {
      result[get_group_id(0)] = buffer[0] + buffer[1];
   }
}

