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

/*
 * Main OpenCL header.
 */
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>

/*
 * Registry for compiled kernels.
 */
typedef struct ocl_kernel_registry {
	char* id_string;
	cl_program program;
	struct ocl_kernel_registry* next;
} ocl_kernel_registry_t;

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
	size_t max_workgroup_size;	/* maximum number of threads per workgrop */
	cl_uint max_compute_units;	/* number of compute processors */
	/* Command queue for this device */
	cl_command_queue queue;
	/* Buffer for result data */
	cl_mem result_buffer;
	size_t result_buffer_size;
	/* Buffer for input data */
	cl_mem input_buffer;
	size_t input_buffer_size;
	/* Kernel registry */
	ocl_kernel_registry_t* kernel_registry;
} ocl_context_t;

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
cl_kernel ocl_getKernel(const char* kernel_name, const char* build_params);

#endif /* USE_OPENCL */
#endif /* OCL_UTILITIES_H_ */

