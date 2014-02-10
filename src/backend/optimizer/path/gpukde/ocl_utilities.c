#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
/*
 * ocl_utilities.c
 *
 *  Created on: 08.05.2012
 *      Author: mheimel
 */

#include "ocl_utilities.h"

#include "catalog/pg_type.h"

#ifdef USE_OPENCL

/*
 * Global context variable.
 */
ocl_context_t* ocl_context = NULL;

/*
 * Global GUC Config variables.
 */
bool ocl_use_gpu;
bool kde_enable;
int kde_samplesize;

/*
 * Accessor for the global context. Make sure to initialize the context on first access.
 */
ocl_context_t* ocl_getContext() {
	if (!ocl_context) ocl_initialize();
	return ocl_context;
}

/*
 * Initialize the global context.
 */
void ocl_initialize(void) {
	cl_int err;
	unsigned int i;

	// Check if the context is already initialized
	if (ocl_context)
		return;

	if (ocl_use_gpu)
		fprintf(stderr, "Initializing OpenCL context for GPU.\n");
	else
		fprintf(stderr, "Initializing OpenCL context for CPU.\n");

	// Get all platform IDs
	cl_uint nr_of_platforms;
	clGetPlatformIDs(0, NULL, &nr_of_platforms);
	if (nr_of_platforms == 0) {
		fprintf(stderr, "No OpenCL platforms found.\n");
		return;
	}
	cl_platform_id platforms[nr_of_platforms];
	clGetPlatformIDs(nr_of_platforms, platforms, NULL);

	// Now select platform and device. We simply pick the first device that is of the requested type.
	cl_platform_id platform = NULL;
	cl_device_id device = NULL;
	for (i=0; i<nr_of_platforms; ++i) {
		cl_uint nr_of_devices;
		err = clGetDeviceIDs(platforms[i], ocl_use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 0, NULL, &nr_of_devices);
		if (!err && nr_of_devices > 0) {
			platform = platforms[i];
			err = clGetDeviceIDs(platform, ocl_use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device, NULL);
			break;
		}
	}

	// Check if we selected a device
	if (device == NULL) {
		fprintf(stderr, "No suitable OpenCL device found.\n");
		return;
	}

	// Allocate a new context and command queue
	ocl_context_t* ctxt = (ocl_context_t*)malloc(sizeof(ocl_context_t));
	memset(ctxt, 0, sizeof(ocl_context_t));
	ctxt->context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	ctxt->device = device;
	ctxt->is_gpu = ocl_use_gpu;
	ctxt->queue = clCreateCommandQueue(ctxt->context, ctxt->device, 0, &err);
	ctxt->program_registry = dictionary_init();

	// Now get some device information parameters:
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &(ctxt->max_alloc_size), NULL);
	err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &(ctxt->global_mem_size), NULL);
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &(ctxt->max_workgroup_size), NULL);
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &(ctxt->max_compute_units), NULL);

	/* Allocate a result buffer of kde_samplesize kB on the device buffer */
	ctxt->result_buffer_size = kde_samplesize * 1024;
	ctxt->result_buffer = clCreateBuffer(ctxt->context, CL_MEM_READ_WRITE,
	                                     ctxt->result_buffer_size, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "\tError allocating OpenCL result buffer.\n");
		goto bad;
	}

	/* Allocate an input buffer for storing input requests */
	ctxt->input_buffer_size = sizeof(float)*10*2;	/* Lower and Upper bound for 10 dimensions */
	ctxt->input_buffer = clCreateBuffer(ctxt->context, CL_MEM_READ_WRITE, ctxt->input_buffer_size, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "\tError allocating OpenCL input buffer.\n");
		goto bad;
	}

	// We are done.
	ocl_context = ctxt;
	fprintf(stderr, "\tOpenCL successfully initialized!\n");
	return;

bad:
	fprintf(stderr, "\tError during OpenCL initialization.\n");
	if (ctxt->queue)
		clReleaseCommandQueue(ctxt->queue);
	if (ctxt->result_buffer)
		clReleaseMemObject(ctxt->result_buffer);
	if (ctxt->context)
		clReleaseContext(ctxt->context);
	if (ctxt)
		free(ctxt);
}

/*
 * Release the global context
 */
void ocl_releaseContext() {
	if (ocl_context == NULL) return;

	fprintf(stderr, "Releasing OpenCL context.\n");

	if (ocl_context->queue) clReleaseCommandQueue(ocl_context->queue);
	// Release kernel registry ressources:
	dictionary_iterator_t it = dictionary_iterator_init(ocl_context->program_registry);
	while (dictionary_iterator_key(ocl_context->program_registry, it)) {
	  clReleaseProgram((cl_program)dictionary_iterator_value(ocl_context->program_registry, it));
	  it = dictionary_iterator_next(ocl_context->program_registry, it);
	}
	dictionary_release(ocl_context->program_registry, 0);
	// Release the result buffer.
	clReleaseMemObject(ocl_context->result_buffer);
	clReleaseContext(ocl_context->context);
	free(ocl_context);
	ocl_context = NULL;
}

// Helper function to read a file into a buffer
static int readFile(FILE* f, char** content, size_t* length) {
   // get file length
   fseek(f, 0, SEEK_END);
   *length = ftell(f);
   rewind(f);

   // allocate buffer
   *content = (char*)malloc(*length + 1);
   if (*content == NULL)
      return 1;

   // now read in the file
   if (fread(*content, *length, 1, f) == 0)
      return 1;
   (*content)[*length] = 0;   // make sure the string is terminated
   return 0;
}

/*
 * List of all kernel names
 */
static const char *kernel_names[] = {
    PGSHAREDIR"/kernel/sum.cl",
    PGSHAREDIR"/kernel/kde.cl",
    PGSHAREDIR"/kernel/init.cl",
    PGSHAREDIR"/kernel/sample_maintenance.cl"};
static const unsigned int nr_of_kernels = 4;

// Helper function to build a program from all kernel files using the given build params
static cl_program buildProgram(ocl_context_t* context, const char* build_params) {
	unsigned int i;
	// Load all kernel files:
	char** file_buffers = (char**)malloc(sizeof(char*)*nr_of_kernels);
	size_t* file_lengths = (size_t*)malloc(sizeof(size_t)*nr_of_kernels);
	for (i = 0; i < nr_of_kernels; ++i) {
		FILE* f = fopen(kernel_names[i], "rb");
		readFile(f, &(file_buffers[i]), &(file_lengths[i]));
		fclose(f);
	}
	// Construct the device-specific build parameters.
	char device_params[2048];
	sprintf(device_params, "-DMAXBLOCKSIZE=%i ", (int)(context->max_workgroup_size));
	if (context->is_gpu) {
		strcat(device_params, "-DDEVICE_GPU ");
	} else {
		strcat(device_params, "-DDEVICE_CPU ");
	}
	strcat(device_params, build_params);
	// Ok, build the program
	cl_program program = clCreateProgramWithSource(
	    context->context, nr_of_kernels,
	    (const char**)file_buffers, (const size_t*)file_lengths, NULL);
	fprintf(stderr, "Compiling OpenCL kernels: %s\n", device_params);
	cl_int err = clBuildProgram(program, 1, &(context->device), device_params, NULL, NULL);
	if (err != CL_SUCCESS) {
		// Print the error log
		fprintf(stderr, "Error compiling the program:\n");
		size_t log_size;
		clGetProgramBuildInfo(program, context->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char* log = malloc(log_size+1);
		clGetProgramBuildInfo(program, context->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		fprintf(stderr, "%s\n", log);
		free(log);
		program = NULL;
		goto cleanup;
	}
	// And add it to the registry:
	dictionary_insert(context->program_registry, build_params, program);

cleanup:
	// Release the resources.
	for (i = 0; i < nr_of_kernels; ++i)
		free(file_buffers[i]);
	free(file_buffers);
	free(file_lengths);
	// We are done.
	return program;
}

/*
 *	Fetches the given kernel for the given build_params.
 */
cl_kernel ocl_getKernel(const char* kernel_name, int dimensions) {
  // We only introduce the number of dimensions into the kernels.
  char build_params[16];
  sprintf(build_params, "-DD=%i", dimensions);
	// Get the context
	ocl_context_t* context = ocl_getContext();
	// Check if we already know the program for the given build_params
	cl_program program = (cl_program)dictionary_get(context->program_registry,
	                                                build_params);
	if (program == NULL) {
		// The program was not found, build a new program using the given build_params.
		program = buildProgram(context, build_params);
		if (program == NULL) {
			return NULL;
		}
	}
	// Ok, we have the program, create the kernel.
	cl_int err;
	cl_kernel result = clCreateKernel(program, kernel_name, &err);
	if (err != CL_SUCCESS) 
		return NULL;
	else
		return result;
}

void ocl_dumpBufferToFile(const char* file, cl_mem buffer,
                          int dimensions, int items) {
  ocl_context_t* context = ocl_getContext();
  clFinish(context->queue);

  // Fetch the buffer to disk.
  float* host_buffer = palloc(sizeof(float) * dimensions * items);
  clEnqueueReadBuffer(context->queue, buffer, CL_TRUE,
                      0, sizeof(float) * dimensions * items,
                      host_buffer, 0, NULL, NULL);

  FILE* f = fopen(file, "w");
  unsigned int i,j;
  for (i=0; i<items; ++i) {
    fprintf(f, "%f", host_buffer[i*dimensions]);
    for (j=1; j<dimensions; ++j) {
      fprintf(f, "; %f", host_buffer[i*dimensions + j]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
  pfree(host_buffer);
}

#endif /* USE_OPENCL */
