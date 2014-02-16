#ifndef TYPE_DEFINED_
  #if (TYPE == float)
    typedef float T;
  #elif (TYPE == double)
    tpyedef double T;
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
   __global T* const result,
   const unsigned int tuples_per_thread
){
   unsigned int local_id = get_local_id(0);
   unsigned int global_id = get_global_id(0);
   // Now each thread does a sequential aggregation within a register.
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

   // Push the local result to the buffer, so we can aggregate recursively. Since the
   // kernel is very simplistic, we can safely assume that it is always run with MAXBLOCKSIZE.
   __local T buffer[MAXBLOCKSIZE];
   buffer[local_id] = agg;
   barrier(CLK_LOCAL_MEM_FENCE);

   // Recursively aggregate the tuples in memory.
   #if (MAXBLOCKSIZE >= 1024)
      if (local_id < 512)
         buffer[local_id] += buffer[local_id + 512];
      barrier(CLK_LOCAL_MEM_FENCE);
   #endif

   #if (MAXBLOCKSIZE >= 512)
      if (local_id < 256)
         buffer[local_id] += buffer[local_id + 256];
      barrier(CLK_LOCAL_MEM_FENCE);
   #endif

   #if (MAXBLOCKSIZE >= 256)
   if (local_id < 128)
         buffer[local_id] += buffer[local_id + 128];
      barrier(CLK_LOCAL_MEM_FENCE);
   #endif

   #if (MAXBLOCKSIZE >= 128)
      if (local_id < 64)
         buffer[local_id] += buffer[local_id + 64];
      barrier(CLK_LOCAL_MEM_FENCE);
   #endif

   // We assume there is a minimum blocksize of 128 threads.
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

