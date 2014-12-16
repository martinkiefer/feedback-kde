#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
/*
 * ocl_utilities.c
 *
 *  Created on: 08.05.2012
 *      Author: mheimel
 */

#include "ocl_utilities.h"

#include <math.h>
#include <sys/time.h>

#include "miscadmin.h"
#include "catalog/pg_type.h"
#include "optimizer/path/gpukde/ocl_estimator_api.h"

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
bool kde_debug;
int kde_samplesize;
int kde_bandwidth_representation;

cl_kernel init_buffer_min = NULL;
cl_kernel init_buffer_sum = NULL;
cl_kernel fast_sum = NULL;
cl_kernel slow_sum = NULL;

cl_kernel fast_min = NULL;
cl_kernel slow_min = NULL;
cl_kernel last_min = NULL;

bool ocl_isDebug() {
  return kde_debug;
}

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
  if (ocl_context) return;
  // Debug output.
  if (ocl_use_gpu) {
    fprintf(stderr, "Initializing OpenCL context for GPU.\n");
  } else {
    fprintf(stderr, "Initializing OpenCL context for CPU.\n");
  }

  // Get all platform IDs
  cl_uint nr_of_platforms;
  clGetPlatformIDs(0, NULL, &nr_of_platforms);
  if (nr_of_platforms == 0) {
    fprintf(stderr, "No OpenCL platforms found.\n");
    return;
  }
  cl_platform_id platforms[nr_of_platforms];
  err = clGetPlatformIDs(nr_of_platforms, platforms, NULL);
  Assert(err == CL_SUCCESS);
  
  // Now select platform and device. We simply pick the first device that is of the requested type.
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  for (i=0; i<nr_of_platforms; ++i) {
    cl_uint nr_of_devices;
    err = clGetDeviceIDs(
        platforms[i], (ocl_use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU),
        0, NULL, &nr_of_devices);
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
  ctxt->queue = clCreateCommandQueue(
      ctxt->context, ctxt->device,
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
  ctxt->program_registry = dictionary_init();
  Assert(err == CL_SUCCESS);
  
  // Now get some device information parameters:
  err |= clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &(ctxt->max_alloc_size), NULL);
  err |= clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &(ctxt->global_mem_size), NULL);
  err |= clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(size_t), &(ctxt->local_mem_size), NULL);
  err |= clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &(ctxt->max_workgroup_size), NULL);
  // Cap to 1024 (our sum kernel cannot deal with larger workgroup sizes atm).
  ctxt->max_workgroup_size = Min(ctxt->max_workgroup_size, 1024);
  err |= clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &(ctxt->max_compute_units), NULL);
  err |= clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &(ctxt->required_mem_alignment), NULL);
  Assert(err == CL_SUCCESS);
  
  // We are done.
  ocl_context = ctxt;
  fprintf(stderr, "\tOpenCL successfully initialized!\n");
  return;

bad:
  fprintf(stderr, "\tError during OpenCL initialization.\n");
  if (ctxt->queue) err = clReleaseCommandQueue(ctxt->queue);
  Assert(err == CL_SUCCESS);
  
  if (ctxt->context) err = clReleaseContext(ctxt->context);
  Assert(err == CL_SUCCESS);
  
  if (ctxt) free(ctxt);
}

/*
 * Release the global context
 */
void ocl_releaseContext() {
  if (ocl_context == NULL) return;
  cl_int err = CL_SUCCESS;
  fprintf(stderr, "Releasing OpenCL context.\n");
  if (ocl_context->queue) err |= clReleaseCommandQueue(ocl_context->queue);
  Assert(err == CL_SUCCESS);
  
  // Release kernel registry ressources:
  dictionary_iterator_t it = dictionary_iterator_init(
      ocl_context->program_registry);
  while (dictionary_iterator_key(ocl_context->program_registry, it)) {
    err = clReleaseProgram((cl_program)dictionary_iterator_value(
        ocl_context->program_registry, it));
    Assert(err == CL_SUCCESS);
    it = dictionary_iterator_next(ocl_context->program_registry, it);
  }
  dictionary_release(ocl_context->program_registry, 0);
  // Release the result buffer.
  err = clReleaseContext(ocl_context->context);
  Assert(err == CL_SUCCESS);
  
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
  if (*content == NULL) return 1;

  // now read in the file
  if (fread(*content, *length, 1, f) == 0) return 1;
  (*content)[*length] = 0;   // make sure the string is terminated
 return 0;
}

/*
 * List of all kernel names
 */
static const char *kernel_names[] = {
    PGSHAREDIR"/kernel/min.cl",
    PGSHAREDIR"/kernel/sum.cl",
    PGSHAREDIR"/kernel/kde.cl",
    PGSHAREDIR"/kernel/init.cl",
    PGSHAREDIR"/kernel/sample_maintenance.cl",
    PGSHAREDIR"/kernel/model_maintenance.cl",
    PGSHAREDIR"/kernel/online_learning.cl"
};
static const unsigned int nr_of_kernels = 7;

static void concatFiles(
    unsigned int nr_of_files, char** file_buffers, size_t* file_lengths,
    char** result_file, size_t* result_file_length) {
  unsigned int i;
  unsigned int wpos;
  // Compute the total length.
  *result_file_length = 0;
  for (i=0; i<nr_of_files; ++i) {
    *result_file_length += file_lengths[i];
  }
  *result_file_length += nr_of_files;
  // Allocate a new buffer.
  *result_file = malloc(*result_file_length);
  wpos = 0;
  for (i=0; i<nr_of_files; ++i) {
    memcpy((*result_file) + wpos, file_buffers[i], file_lengths[i]);
    wpos += file_lengths[i];
    (*result_file)[wpos++] = '\n';
  }
}

// Helper function to build a program from all kernel files using the given build params
static cl_program buildProgram(
    ocl_context_t* context, const char* build_params) {
  unsigned int i;
  // Load all kernel files:
  char** file_buffers = (char**)malloc(sizeof(char*)*nr_of_kernels);
  size_t* file_lengths = (size_t*)malloc(sizeof(size_t)*nr_of_kernels);
  for (i = 0; i < nr_of_kernels; ++i) {
    FILE* f = fopen(kernel_names[i], "rb");
    readFile(f, &(file_buffers[i]), &(file_lengths[i]));
    fclose(f);
  }
  // Concatenate all kernels into a single buffer.
  char* kernel_sources;
  size_t kernel_source_length;
  concatFiles(
      nr_of_kernels, file_buffers, file_lengths,
      &kernel_sources, &kernel_source_length);
  // Write the sources to a debug file (if requested).
  char kernel_source_file_name[1024];
  if (ocl_isDebug()) {
    sprintf(kernel_source_file_name, "%s/kernel_debug.cl", DataDir);
    FILE* kernel_source_file = fopen(kernel_source_file_name, "w");
    fwrite(kernel_sources, kernel_source_length, 1, kernel_source_file);
    fclose(kernel_source_file);
  }
  // Construct the device-specific build parameters.
  char device_params[1024];
  sprintf(
      device_params, "-DMAXBLOCKSIZE=%i ", (int)(context->max_workgroup_size));
  if (context->is_gpu) {
    strcat(device_params, "-DDEVICE_GPU ");
  } else {
    strcat(device_params, "-DDEVICE_CPU ");
    if (ocl_isDebug()) {
      // Add the required debug flags.
      strcat(device_params, "-g -s \"");
      strcat(device_params, kernel_source_file_name);
      strcat(device_params, "\" ");
    }
  }
  if (sizeof(kde_float_t) == sizeof(double)) {
    strcat(device_params, "-DTYPE=8 ");
  } else if (sizeof(kde_float_t) == sizeof(float)) {
    strcat(device_params, "-DTYPE=4 ");
  }
  strcat(device_params, build_params);

  // Ok, build the program.
  cl_int err = CL_SUCCESS;
  cl_program program = clCreateProgramWithSource(
      context->context, 1, (const char**)&kernel_sources,
      &kernel_source_length, &err);
  Assert(err == CL_SUCCESS);
  
  fprintf(stderr, "Compiling OpenCL kernels %s\n", device_params);
  err = clBuildProgram(
      program, 1, &(context->device), device_params, NULL, NULL);
  if (err != CL_SUCCESS) {
    // Print the error log
    fprintf(stderr, "Error compiling the program:\n");
    size_t log_size;
    err = clGetProgramBuildInfo(
        program, context->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    Assert(err == CL_SUCCESS);
    
    char* log = malloc(log_size+1);
    err = clGetProgramBuildInfo(
        program, context->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    fprintf(stderr, "%s\n", log);
    Assert(err == CL_SUCCESS);
    
    free(log);
    program = NULL;
    goto cleanup;
  }
  // And add it to the registry:
  dictionary_insert(context->program_registry, build_params, program);

cleanup:
  // Release the resources.
  for (i = 0; i < nr_of_kernels; ++i) {
    free(file_buffers[i]);
  }
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
  char build_params[32];
  
  if(kde_bandwidth_representation == LOG_BW){
    sprintf(build_params, "-DLOG_BANDWIDTH -DD=%i",dimensions);
  }
  else {
    sprintf(build_params, "-DD=%i", dimensions);
  }  

  // Get the context
  ocl_context_t* context = ocl_getContext();
  // Check if we already know the program for the given build_params
  cl_program program = (cl_program)dictionary_get(
      context->program_registry, build_params);
  if (program == NULL) {
    // The program was not found, build a new program using the given build_params.
    program = buildProgram(context, build_params);
    if (program == NULL) return NULL;
  }
  // Ok, we have the program, create the kernel.
  cl_int err;
  cl_kernel result = clCreateKernel(program, kernel_name, &err);
  if (err != CL_SUCCESS) {
    return NULL;
  } else {
    return result;
  }
}

void ocl_dumpBufferToFile(
    const char* file, cl_mem buffer, int dimensions, int items) {

  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  clFinish(context->queue);
  // Fetch the buffer to disk.
  kde_float_t* host_buffer = palloc(sizeof(kde_float_t) * dimensions * items);
  err = clEnqueueReadBuffer(
      context->queue, buffer, CL_TRUE, 0,
      sizeof(kde_float_t) * dimensions * items, host_buffer, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  FILE* f = fopen(file, "w");
  unsigned int i,j;
  for (i=0; i<items; ++i) {
    fprintf(f, "%f", host_buffer[i*dimensions]);
    for (j=1; j<dimensions; ++j) {
      fprintf(f, "\t%f", host_buffer[i*dimensions + j]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
  pfree(host_buffer);
}

void ocl_printBuffer(
    const char* message, cl_mem buffer, int dimensions, int items) {
  if (!kde_debug) return;
  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  clFinish(context->queue);  // Make sure that all changes are materialized.
  unsigned int i,j;
  // Fetch the buffer to the host.
  kde_float_t* host_buffer = palloc(sizeof(kde_float_t) * dimensions * items);
  err = clEnqueueReadBuffer(
      context->queue, buffer, CL_TRUE, 0,
      sizeof(kde_float_t) * dimensions * items, host_buffer, 0, NULL, NULL);
  Assert(err == CL_SUCCESS);
  
  fprintf(stderr, "%s", message);
  if (items == 1) {
    fprintf(stderr, " ");
    for (i=0; i<dimensions; ++i) {
      fprintf(stderr, "%E ", host_buffer[i]);
    }
  } else {
    fprintf(stderr, "\n\t");
    for (i=0; i<items; ++i) {
      for (j=0; j<dimensions; ++j) {
        fprintf(stderr, "%E ", host_buffer[i*dimensions + j]);
      }
      fprintf(stderr, "\n\t");
    }
  }
  fprintf(stderr, "\n");
  pfree(host_buffer);
}

ocl_aggregation_descriptor_t* prepareSumDescriptor(
    cl_mem input_buffer, unsigned int elements,
    cl_mem result_buffer, unsigned int result_buffer_offset) {
  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  
  ocl_aggregation_descriptor_t* descriptor = calloc(
      1, sizeof(ocl_aggregation_descriptor_t));
  // Prepare the kernels.
  descriptor->pre_aggregation = ocl_getKernel("sum_par", 0);
  descriptor->final_aggregation = ocl_getKernel("sum_seq", 0);
  // Determine the optimal local size.
  err = clGetKernelWorkGroupInfo(
        descriptor->pre_aggregation, context->device, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(size_t), &(descriptor->local_size), NULL);
  Assert(err == CL_SUCCESS);
  
  // Truncate to local memory requirements.
  descriptor->local_size = Min(
      descriptor->local_size, context->local_mem_size / sizeof(kde_float_t));
  // And truncate to the next power of two.
  descriptor->local_size =
      (size_t)0x1 << (int)(log2((double)descriptor->local_size));
  // Allocate the temporary result buffer.
  descriptor->intermediate_result_buffer =  clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * context->max_compute_units, NULL, &err);
  Assert(err == CL_SUCCESS);
  
  // Figure out how many elements we have to aggregate per thread:
  unsigned int tuples_per_thread =
      elements / (context->max_compute_units * descriptor->local_size);
  if (tuples_per_thread * (context->max_compute_units * descriptor->local_size) < elements) {
    tuples_per_thread++;
  }
  // Prepare the pre-aggregation kernel.
  err |= clSetKernelArg(
      descriptor->pre_aggregation, 0, sizeof(cl_mem), &input_buffer);
  err |= clSetKernelArg(
      descriptor->pre_aggregation, 1,
      sizeof(kde_float_t) * descriptor->local_size, NULL);
  err |= clSetKernelArg(
      descriptor->pre_aggregation, 2, sizeof(cl_mem),
      &(descriptor->intermediate_result_buffer));
  err |= clSetKernelArg(
      descriptor->pre_aggregation, 3, sizeof(unsigned int), &tuples_per_thread);
  err |= clSetKernelArg(
      descriptor->pre_aggregation, 4,
      sizeof(unsigned int), &elements);
  Assert(err == CL_SUCCESS);
  
  // Prepare the post-aggregation kernel.
  unsigned int zero = 0;
  err |= clSetKernelArg(
      descriptor->final_aggregation, 0, sizeof(cl_mem),
      &(descriptor->intermediate_result_buffer));
  err |= clSetKernelArg(
      descriptor->final_aggregation, 1, sizeof(unsigned int), &zero);
  err |= clSetKernelArg(
      descriptor->final_aggregation, 2, sizeof(unsigned int),
      &(context->max_compute_units));
  err |= clSetKernelArg(
      descriptor->final_aggregation, 3, sizeof(cl_mem), &result_buffer);
  err |= clSetKernelArg(
      descriptor->final_aggregation, 4, sizeof(unsigned int),
      &result_buffer_offset);
  Assert(err == CL_SUCCESS);
  
  // We are done :)
  return descriptor;
}

void releaseAggregationDescriptor(ocl_aggregation_descriptor_t* descriptor) {
  cl_int err = CL_SUCCESS;
  if (descriptor->intermediate_result_buffer) {
    err = clReleaseMemObject(descriptor->intermediate_result_buffer);
    Assert(err == CL_SUCCESS);
  }
  if (descriptor->pre_aggregation) {
    err = clReleaseKernel(descriptor->pre_aggregation);
    Assert(err == CL_SUCCESS);
  }
  if (descriptor->final_aggregation) {
    err = clReleaseKernel(descriptor->final_aggregation);
    Assert(err == CL_SUCCESS);
  }
  free(descriptor);
}

cl_event predefinedSumOfArray(
    ocl_aggregation_descriptor_t* sum_descriptor, cl_event external_event) {
  ocl_context_t* context = ocl_getContext();
  cl_int err = CL_SUCCESS;
  size_t processors = context->max_compute_units;
  // Schedule the pre-aggregation.
  size_t global_size = sum_descriptor->local_size * processors;
  cl_event pre_aggregation_event;
  if (external_event) {
    err = clEnqueueNDRangeKernel(
        context->queue, sum_descriptor->pre_aggregation, 1, NULL, &global_size,
        &(sum_descriptor->local_size), 1, &external_event,
        &pre_aggregation_event);
    Assert(err == CL_SUCCESS);
  } else {
    err = clEnqueueNDRangeKernel(
        context->queue, sum_descriptor->pre_aggregation, 1, NULL, &global_size,
        &(sum_descriptor->local_size), 0, NULL, &pre_aggregation_event);
  }
  // Now perform a final pass over the data to compute the aggregate.
  global_size = 1;
  cl_event finalize_event;
  err = clEnqueueNDRangeKernel(
      context->queue, sum_descriptor->final_aggregation, 1, NULL, &global_size,
      NULL, 1, &pre_aggregation_event, &finalize_event);
  Assert(err == CL_SUCCESS);
  
  err = clReleaseEvent(pre_aggregation_event);
  Assert(err == CL_SUCCESS);
  
  return finalize_event;
}

cl_event sumOfArray(
    cl_mem input_buffer, unsigned int elements,
    cl_mem result_buffer, unsigned int result_buffer_offset,
    cl_event external_event) {
  ocl_aggregation_descriptor_t* desc = prepareSumDescriptor(
      input_buffer, elements, result_buffer, result_buffer_offset);
  cl_event result_event = predefinedSumOfArray(desc, external_event);
  releaseAggregationDescriptor(desc);
  return result_event;
}

cl_event minOfArray(
    cl_mem input_buffer, unsigned int elements,
    cl_mem result_min,cl_mem result_index, unsigned int result_buffer_offset,
    cl_event external_event) {
  cl_int err = CL_SUCCESS;
  ocl_context_t* context = ocl_getContext();

  cl_event events[] = { NULL, NULL };
  unsigned int nr_of_events = 0;
  
  // Fetch the required sum kernels:
  if (init_buffer_min == NULL) init_buffer_min = ocl_getKernel("init", 0);
  if (fast_min == NULL) fast_min = ocl_getKernel("min_par", 0);
  if (slow_min == NULL) slow_min = ocl_getKernel("min_seq", 0);
  if (last_min == NULL) last_min = ocl_getKernel("min_seq_last_pass", 0);
  struct timeval start; gettimeofday(&start, NULL);

  // Determine the optimal local size.
  size_t local_size;
  err = clGetKernelWorkGroupInfo(
      fast_sum, context->device, CL_KERNEL_WORK_GROUP_SIZE,
      sizeof(size_t), &local_size, NULL);
  Assert(err == CL_SUCCESS);
  
  // Truncate to local memory requirements, we need two local floating points per thread
  local_size = Min(
      local_size,
      context->local_mem_size / (sizeof(kde_float_t) * 2));
  // And truncate to the next power of two.
  local_size = (size_t)0x1 << (int)(log2((double)local_size));

  size_t processors = context->max_compute_units;

  // Figure out how many elements we can aggregate per thread in the parallel part:
  unsigned int tuples_per_thread = elements / (processors * local_size);
  // Now compute the configuration of the sequential kernel:
  unsigned int slow_kernel_data_offset = processors * tuples_per_thread * local_size;
  unsigned int slow_kernel_elements = elements - slow_kernel_data_offset;
  unsigned int slow_kernel_result_offset = processors;

  // Allocate a temporary result buffers.
  cl_mem tmp_buffer_min = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(kde_float_t) * (processors + 1), NULL, &err);
  Assert(err == CL_SUCCESS);
  
  cl_mem tmp_buffer_index = clCreateBuffer(
      context->context, CL_MEM_READ_WRITE,
      sizeof(unsigned int) * (processors + 1), NULL, &err);
  Assert(err == CL_SUCCESS);
  
  size_t global_size = processors + 1; 
  
  //Preinitialize the value buffer with +INNFINITY. 
  kde_float_t inf = INFINITY;
  err |= clSetKernelArg(init_buffer_min, 0, sizeof(cl_mem), &tmp_buffer_min);
  err |= clSetKernelArg(init_buffer_min, 1, sizeof(kde_float_t), &inf);
  Assert(err == CL_SUCCESS);
  
  cl_event init_event;
  err = clEnqueueNDRangeKernel(
      context->queue, init_buffer_min, 1, NULL, &global_size,
      NULL, 0, NULL, &init_event);
  Assert(err == CL_SUCCESS);
  
  // Ok, we selected the correct kernel and parameters. Now prepare the arguments.
  err |= clSetKernelArg(fast_min, 0, sizeof(cl_mem), &input_buffer);
  err |= clSetKernelArg(fast_min, 1, sizeof(kde_float_t) * local_size, NULL);
  err |= clSetKernelArg(fast_min, 2, sizeof(kde_float_t) * local_size, NULL);
  err |= clSetKernelArg(fast_min, 3, sizeof(cl_mem), &tmp_buffer_min);
  err |= clSetKernelArg(fast_min, 4, sizeof(cl_mem), &tmp_buffer_index);
  err |= clSetKernelArg(fast_min, 5, sizeof(unsigned int), &tuples_per_thread);
  err |= clSetKernelArg(slow_min, 0, sizeof(cl_mem), &input_buffer);
  err |= clSetKernelArg(slow_min, 1, sizeof(unsigned int), &slow_kernel_data_offset);
  err |= clSetKernelArg(slow_min, 2, sizeof(unsigned int), &slow_kernel_elements);
  err |= clSetKernelArg(slow_min, 3, sizeof(cl_mem), &tmp_buffer_min);
  err |= clSetKernelArg(slow_min, 4, sizeof(cl_mem), &tmp_buffer_index);
  err |= clSetKernelArg(slow_min, 5, sizeof(unsigned int), &slow_kernel_result_offset);
  Assert(err == CL_SUCCESS);
  
  // Fire the kernel
  if (tuples_per_thread) {
    global_size = local_size * processors;
    cl_event event;
    err = clEnqueueNDRangeKernel(
        context->queue, fast_min, 1, NULL, &global_size,
        &local_size, 1, &init_event, &event);
    Assert(err == CL_SUCCESS);
    events[nr_of_events++] = event;
  }
  if (slow_kernel_elements) {
    cl_event event;
    err = clEnqueueTask(context->queue, slow_min, 1, &init_event, &event);
    Assert(err == CL_SUCCESS);
    events[nr_of_events++] = event;
  }

  // Now perform a final pass over the data to compute the aggregate.
  slow_kernel_data_offset = 0;
  slow_kernel_elements = processors + 1;
  slow_kernel_result_offset = 0;
  err = 0;
  err |= clSetKernelArg(last_min, 0, sizeof(cl_mem), &tmp_buffer_min);
  err |= clSetKernelArg(last_min, 1, sizeof(cl_mem), &tmp_buffer_index);
  err |= clSetKernelArg(last_min, 2, sizeof(unsigned int), &slow_kernel_data_offset);
  err |= clSetKernelArg(last_min, 3, sizeof(unsigned int), &slow_kernel_elements);
  err |= clSetKernelArg(last_min, 4, sizeof(cl_mem), &result_min);
  err |= clSetKernelArg(last_min, 5, sizeof(cl_mem), &result_index);
  err |= clSetKernelArg(last_min, 6, sizeof(unsigned int), &result_buffer_offset);  
  Assert(err == CL_SUCCESS);
  
  cl_event finalize_event;
  err = clEnqueueTask(context->queue, last_min, nr_of_events, events, &finalize_event);
  Assert(err == CL_SUCCESS);
  
  // Clean up ...
  err |= clReleaseMemObject(tmp_buffer_min);
  err |= clReleaseMemObject(tmp_buffer_index);
  Assert(err == CL_SUCCESS);
  
  if (events[0]) err |= clReleaseEvent(events[0]);
  if (events[1]) err |= clReleaseEvent(events[1]);
  Assert(err == CL_SUCCESS);
  
  return finalize_event;
}

#endif /* USE_OPENCL */
