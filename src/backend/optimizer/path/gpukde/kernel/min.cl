#pragma OPENCL EXTENSION cl_khr_fp64: enable

#ifndef TYPE_DEFINED_
  #if (TYPE == 4)
    typedef float T;
  #elif (TYPE == 8)
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    typedef double T;
  #endif
  #define TYPE_DEFINED_
#endif /* TYPE_DEFINED */
    
__kernel void min_seq(
   __global const T* const data,
   const unsigned int data_offset,
   const unsigned int elements,
   __global T* const result_min,
   __global unsigned int* const result_index,
   const unsigned int result_offset
){
   T min = INFINITY;
   unsigned int index = 0;
   for (unsigned i=0; i<elements; ++i){
	unsigned int element = data_offset + i; 
	if(data[element] < min){
	    index = element;
	    min = data[element];
	 }
   }
   result_min[result_offset] = min;
   result_index[result_offset] = index;
}

__kernel void min_seq_last_pass(
   __global const T* const data_min,
   __global const unsigned int* const data_index,
   const unsigned int data_offset,
   const unsigned int elements,
   __global T* const result_min,
   __global unsigned int* const result_index,
   const unsigned int result_offset
){
   T min = INFINITY;
   unsigned int index = 0;
   for (unsigned i=0; i<elements; ++i){
	unsigned int element = data_offset + i; 
	if(data_min[element] < min){
	    index = data_index[element];
	    min = data_min[element];
	 }
   }
   result_min[result_offset] = min;
   result_index[result_offset] = index;
}

__kernel void min_par(
   __global const T* const data,
   __local T* buffer_min,
   __local unsigned int* buffer_index,
   __global T* const result_min,
   __global unsigned int* const result_index,
   const unsigned int tuples_per_thread
){
   unsigned int local_id = get_local_id(0);
   unsigned int global_id = get_global_id(0);
   // Each thread first does a sequential min within a register.
   T min = INFINITY;
   unsigned int index = 0;
   #ifdef DEVICE_GPU
      // On the GPU we use a strided access pattern, so that the GPU
      // can coalesc memory access.
      unsigned int group_start = get_local_size(0)*tuples_per_thread*get_group_id(0);
      for (unsigned int i=0; i < tuples_per_thread; ++i){
	unsigned int element = group_start + i*get_local_size(0) + local_id;
         if(data[element] < min){
	    index = element;
	    min = data[element];
	 }
      }
   #elif defined DEVICE_CPU
      // On the CPU we use a sequential access pattern to keep cache misses
      // local per thread.
      for (unsigned int i=0; i < tuples_per_thread; ++i){
	unsigned int element = tuples_per_thread*global_id + i;
         if(data[element] < min){
	    index = element;
	    min = data[element];
	 }
      }
   #endif

   // Push the local result to the local buffer, so we can aggregate the remainder
   // recursively.
   buffer_min[local_id] = min;
   buffer_index[local_id] = index;
   barrier(CLK_LOCAL_MEM_FENCE);

   // Recursively aggregate the tuples in memory.
   if (get_local_size(0) >= 4096) {
      if (local_id < 2048){
	    if(buffer_min[local_id] > buffer_min[local_id + 2048]){
		buffer_min[local_id] = buffer_min[local_id + 2048];
		buffer_index[local_id] = buffer_index[local_id +2048];
	    }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   // Recursively aggregate the tuples in memory.
   if (get_local_size(0) >= 2048) {
      if (local_id < 1024){
	    if(buffer_min[local_id] > buffer_min[local_id + 1024]){
		buffer_min[local_id] = buffer_min[local_id + 1024];
		buffer_index[local_id] = buffer_index[local_id + 1024];
	    }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 1024) {
      if (local_id < 512){
	    if(buffer_min[local_id] > buffer_min[local_id + 512]){
		buffer_min[local_id] = buffer_min[local_id + 512];
		buffer_index[local_id] = buffer_index[local_id + 512];
	    }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 512) {
      if (local_id < 256){
	    if(buffer_min[local_id] > buffer_min[local_id + 256]){
		buffer_min[local_id] = buffer_min[local_id + 256];
		buffer_index[local_id] = buffer_index[local_id + 256];
	    }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 512) {
      if (local_id < 256){
	    if(buffer_min[local_id] > buffer_min[local_id + 256]){
		buffer_min[local_id] = buffer_min[local_id + 256];
		buffer_index[local_id] = buffer_index[local_id + 256];
	    }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 256) {
      if (local_id < 128){
	    if(buffer_min[local_id] > buffer_min[local_id + 128]){
		buffer_min[local_id] = buffer_min[local_id + 128];
		buffer_index[local_id] = buffer_index[local_id + 128];
	    }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if (get_local_size(0) >= 128) {
      if (local_id < 64){
	    if(buffer_min[local_id] > buffer_min[local_id + 64]){
		buffer_min[local_id] = buffer_min[local_id + 64];
		buffer_index[local_id] = buffer_index[local_id + 64];
	    }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }
   
   if (local_id < 32){
	if(buffer_min[local_id] > buffer_min[local_id + 32]){
	    buffer_min[local_id] = buffer_min[local_id + 32];
	    buffer_index[local_id] = buffer_index[local_id + 32];
	}
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   
   
   if (local_id < 16){
	if(buffer_min[local_id] > buffer_min[local_id + 16]){
	    buffer_min[local_id] = buffer_min[local_id + 16];
	    buffer_index[local_id] = buffer_index[local_id + 16];
	}
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   if (local_id < 8){
	if(buffer_min[local_id] > buffer_min[local_id + 8]){
	    buffer_min[local_id] = buffer_min[local_id + 8];
	    buffer_index[local_id] = buffer_index[local_id + 8];
	}
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   if (local_id < 4){
	if(buffer_min[local_id] > buffer_min[local_id + 4]){
	    buffer_min[local_id] = buffer_min[local_id + 4];
	    buffer_index[local_id] = buffer_index[local_id + 4];
	}
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   if (local_id < 2){
	if(buffer_min[local_id] > buffer_min[local_id + 2]){
	    buffer_min[local_id] = buffer_min[local_id + 2];
	    buffer_index[local_id] = buffer_index[local_id + 2];
	}
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   // Ok, we are done, write the result back.
   if (local_id == 0) {
	if(buffer_min[0] > buffer_min[1]){
	    buffer_min[0] = buffer_min[1];
	    buffer_index[0] = buffer_index[1];
	}
   
  }
  
  if (local_id == 0) {
      if(buffer_min[0] > buffer_min[1]){
	    buffer_min[0] = buffer_min[1];
	    buffer_index[0] = buffer_index[1];
      }
      result_min[get_group_id(0)] = buffer_min[0];
      result_index[get_group_id(0)] = buffer_index[1];
   }
}
