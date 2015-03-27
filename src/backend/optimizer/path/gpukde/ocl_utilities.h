/*
 * oclinit.h
 *
 *  Initialization routines for OpenCL.
 *
 *  Created on: 08.05.2012
 *      Author: mheimel
 */

#ifndef OCL_UTILITIES_H_
#define OCL_UTILITIES_H_

// Required postgres imports
#include "postgres.h"
#include "../port/pg_config_paths.h"

#ifdef USE_OPENCL

#include <sys/time.h>

#include "container/dictionary.h"

/*
 * Main OpenCL header.
 */
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>

/*
 * Define whether we use single or double precision.
 */
typedef double kde_float_t;

/*
 * Struct defining the current OpenCL context
 */
typedef struct {
	cl_context context;
	/* Some information about the selected device */
	cl_device_id device;
	cl_bool is_gpu;
	size_t max_alloc_size;		/* largest possible allocation */
	size_t global_mem_size; 	/* global memory size */
	size_t local_mem_size;    /* local memory size */
	size_t max_workgroup_size;	/* maximum number of threads per workgrop */
	cl_uint max_compute_units;	/* number of compute processors */
	cl_uint required_mem_alignment; /* required memory alignment in bits */
	/* Command queue for this device */
	cl_command_queue queue;
	/* Kernel registry */
	dictionary_t program_registry; // Keeps a mapping from build parameters to OpenCL programs.
} ocl_context_t;

// #########################################################################
// ################## FUNCTIONS FOR DEBUGGING ##############################
/**
 * ocl_printBufferToFile / ocl_printBuffer
 *
 * Takes the content of the given buffer and writes it to the provided file.
 * The function assumes that the buffer is of size dimensions*items*sizeof(float)
 *
 * Note: Both functions only work if kde_debug is set.
 */

void ocl_dumpBufferToFile(
    const char* file, cl_mem buffer, int dimensions, int items);
void ocl_printBuffer(
    const char* message, cl_mem buffer, int dimensions, int items);

// Returns true if debugging is enabled.
bool ocl_isDebug(void);

// #########################################################################
// ################## FUNCTIONS FOR OCL CONTEXT MANAGEMENT #################

/*
 * initializeOpenCLContext:
 * Helper function to initialize the context for OpenCL.
 * The function takes a single argument "use_gpu", whic
 * determines the chosen device.
 */
void ocl_initialize(void);

/*
 * getOpenCLContext:
 * Helper function to get the current OpenCL context.
 */
ocl_context_t* ocl_getContext(void);

/*
 * releaseOpenClContext:
 * Helper function to release a given OpenCL context.
 */
void ocl_releaseContext(void);

// #########################################################################
// ################## FUNCTIONS FOR DYNAMIC KERNEL BINDING #################

/*
 * Get an instance of the given kernel using the given build_params.
 */
cl_kernel ocl_getKernel(const char* kernel_name, int dimensions);

// #########################################################################
// ############## HELPER FUNCTIONS FOR THE COMPUTATIONS ####################

typedef struct ocl_aggregation_descriptor {
  // Temporary buffer.
  cl_mem intermediate_result_buffer;
  // Call sizes.
  size_t local_size;
  // Required kernels.
  cl_kernel pre_aggregation;
  cl_kernel final_aggregation;
} ocl_aggregation_descriptor_t;

// Prepares a sum aggregation descriptor for the given buffers.
ocl_aggregation_descriptor_t* prepareSumDescriptor(
    cl_mem input_buffer, unsigned int elements,
    cl_mem result_buffer, unsigned int result_buffer_offset);

// Release the aggregation descriptor.
void releaseAggregationDescriptor(ocl_aggregation_descriptor_t* descriptor);
/*
 * Computes the sum of the elements in input_buffer, writing it to the
 * specified position in result_buffer.
 */
cl_event predefinedSumOfArray(
    ocl_aggregation_descriptor_t* descriptor, cl_event external_event);

// Helper function to compute the sum of an array.
cl_event sumOfArray(
    cl_mem input_buffer, unsigned int elements,
    cl_mem result_buffer, unsigned int result_buffer_offset,
    cl_event external_event);

/*
 * Computes the min of the elements in input_buffer, writing minimum and value
 * to the specified position in result_* buffers.
 */
cl_event minOfArray(
    cl_mem input_buffer, unsigned int elements,
    cl_mem result_min, cl_mem result_index,
    unsigned int result_buffer_offset, cl_event external_event);

// Macros for timing operations.
FILE* kde_getTimingFile(void);

#define CREATE_TIMER() struct timeval __start; gettimeofday(&__start, NULL);
#define RESET_TIMER() gettimeofday(&__start, NULL);
#define READ_TIMER(variable) if (kde_getTimingFile()) {      \
   struct timeval __now; gettimeofday(&__now, NULL);         \
   long __seconds = __now.tv_sec - __start.tv_sec;           \
   long __useconds = __now.tv_usec - __start.tv_usec;        \
   long long __time = (__seconds) * 1000000 + __useconds;    \
   variable = __time;                                        \
}
#define LOG_TIME(text, time) if (kde_getTimingFile()) {      \
   FILE* __f = kde_getTimingFile();                          \
   fprintf(__f, text ": %lld microseconds.\n", time);        \
   fflush(__f);                                              \
}
#define LOG_TIMER(text) if (kde_getTimingFile()) {           \
   long long ___time = 0;                                    \
   READ_TIMER(___time);                                      \
   LOG_TIME(text, ___time);                                  \
}

#endif /* USE_OPENCL */
#endif /* OCL_UTILITIES_H_ */

