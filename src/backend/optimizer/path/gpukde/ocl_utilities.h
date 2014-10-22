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
	/* Buffer for result data */
	cl_mem result_buffer;
	size_t result_buffer_size;
	/* Buffer for input data */
	cl_mem input_buffer;
	size_t input_buffer_size;
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

void ocl_dumpBufferToFile(const char* file, cl_mem buffer, int dimensions, int items);
void ocl_printBuffer(const char* message, cl_mem buffer, int dimensions, int items);

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

/*
 * Computes the sum of the elements in input_buffer, writing it to the
 * specified position in result_buffer.
 *
 */
cl_event sumOfArray(cl_mem input_buffer, unsigned int elements,
                    cl_mem result_buffer, unsigned int result_buffer_offset,
                    cl_event external_event);
/*
 * Computes the min of the elements in input_buffer, writing minimum and value
 * to the specified position in result_* buffers.
 *
 */
cl_event minOfArray(cl_mem input_buffer, unsigned int elements,
                    cl_mem result_min, cl_mem result_index,
                    unsigned int result_buffer_offset, cl_event external_event);

#endif /* USE_OPENCL */
#endif /* OCL_UTILITIES_H_ */

