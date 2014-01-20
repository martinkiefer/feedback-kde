/*
 * estimator.c
 *
 *  Created on: 25.05.2012
 *      Author: mheimel
 */

#include "ocl_estimator.h"
#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
#ifdef USE_OPENCL

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#include "../include/catalog/pg_type.h"

extern bool ocl_use_gpu;
extern bool enable_kde_estimator;
extern int kde_samplesize;

/*
 * Global registry variable
 */
ocl_estimator_registry_t* registry = NULL;

static float sumOfArray(cl_mem input_buffer, cl_mem result_buffer, unsigned int elements) {
	ocl_context_t* context = ocl_getContext();
	// Fetch the required sum kernels:
	cl_kernel fast_sum = ocl_getKernel("sum_par", "-DD=1 -DSAMPLE_SIZE=1");
	cl_kernel slow_sum = ocl_getKernel("sum_seq", "-DD=1 -DSAMPLE_SIZE=1");
	clFinish(context->queue);
	struct timeval start; gettimeofday(&start, NULL);
	// Determine the kernel parameters:
	size_t local_size = context->max_workgroup_size;
	size_t processors = context->max_compute_units;
	// Figure out how many elements we can aggregate per thread in the parallel part:
	unsigned int tuples_per_thread = elements / (processors * local_size);
	// Now compute the configuration of the sequential kernel:
	unsigned int slow_kernel_data_offset = processors * tuples_per_thread * local_size;
	unsigned int slow_kernel_elements = elements - slow_kernel_data_offset;
	unsigned int slow_kernel_result_offset = processors;
	// Ok, we selected the correct kernel and parameters. Now prepare the arguments.
	cl_int err = 0;
	err |= clSetKernelArg(fast_sum, 0, sizeof(cl_mem), &input_buffer);
	err |= clSetKernelArg(fast_sum, 1, sizeof(cl_mem), &result_buffer);
	err |= clSetKernelArg(fast_sum, 2, sizeof(unsigned int), &tuples_per_thread);
	err |= clSetKernelArg(slow_sum, 0, sizeof(cl_mem), &input_buffer);
	err |= clSetKernelArg(slow_sum, 1, sizeof(unsigned int), &slow_kernel_data_offset);
	err |= clSetKernelArg(slow_sum, 2, sizeof(unsigned int), &slow_kernel_elements);
	err |= clSetKernelArg(slow_sum, 3, sizeof(cl_mem), &result_buffer);
	err |= clSetKernelArg(slow_sum, 4, sizeof(unsigned int), &slow_kernel_result_offset);
	// Fire the kernel
	if (tuples_per_thread) {
		size_t global_size = local_size * processors;
		err |= clEnqueueNDRangeKernel(context->queue, fast_sum, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	}
	if (slow_kernel_elements)
		err |= clEnqueueTask(context->queue, slow_sum, 0, NULL, NULL);
	clEnqueueBarrier(context->queue);
	// Now transfer the partial aggregates back to a local buffer ...
	float* partial_aggregates = (float*)malloc(sizeof(float)*(1+processors));
	err |= clEnqueueReadBuffer(context->queue, result_buffer, CL_TRUE, 0, (1+processors)*sizeof(float), partial_aggregates, 0, NULL, NULL);
	// .. and compute the final aggregate.
	float result = 0;
	if (tuples_per_thread) {
		result = partial_aggregates[0];
		unsigned int i;
		for (i = 1; i < processors; ++i)
			result += partial_aggregates[i];
	}
	if (slow_kernel_elements)
		result += partial_aggregates[processors];
	// Clean up ...
	clReleaseKernel(slow_sum);
	clReleaseKernel(fast_sum);
	free(partial_aggregates);
	// .. and return the result.
	struct timeval now; gettimeofday(&now, NULL); long seconds = now.tv_sec - start.tv_sec; long useconds = now.tv_usec - start.tv_usec;
	long mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
	fprintf(stderr, "Sum: %li ms\n", mtime);
	return result;
}

static float rangeKDE(ocl_context_t* ctxt, ocl_estimator_t* estimator) {
	clEnqueueNDRangeKernel(ctxt->queue, estimator->range_kernel, 1, NULL, &(estimator->sample_size), NULL, 0, NULL, NULL);
	clEnqueueBarrier(ctxt->queue);
	return sumOfArray(ctxt->result_buffer, ctxt->result_buffer, estimator->sample_size) * estimator->range_normalization_factor;
}

/* Custom comparator for a table request */
static int compareEstimator(const void* a, const void* b) {
	if ( *(Oid*)a > *(Oid*)b )
		return 1;
	else if  ( *(Oid*)a == *(Oid*)b )
		return 0;
	else
		return -1;
}

/*
 * Initialize the registry.
 */
static void ocl_initializeRegistry() {
	if (registry)
		return;
	registry = malloc(sizeof(ocl_estimator_registry_t));
	memset(registry, 0, sizeof(ocl_estimator_registry_t));
	// Initialize a sample estimator
	//TODO: Load existing estimators from disk.
	//measurePerformance(&(registry->perf_stats));
}

/* 
 *  Static helper function to release the resources held by a single estimator.
 */
static void ocl_freeEstimator(ocl_estimator_t* estimator) {
	if (estimator->column_numbers)
		free(estimator->column_numbers);
	if (estimator->scale_factors)
		free(estimator->scale_factors);
	if (estimator->data_sample)
		clReleaseMemObject(estimator->data_sample);
	if (estimator->bandwidth_buffer)
		clReleaseMemObject(estimator->bandwidth_buffer);
	if (estimator->range_kernel)
		clReleaseKernel(estimator->range_kernel);
}

static void ocl_releaseRegistry() {
	if (!registry)
		return;
	unsigned int i;
	for (i=0; i<registry->nr_of_estimators; ++i) {
		ocl_freeEstimator(&(registry->estimators[i]));	
	}
	free(registry);
	registry = NULL;	
}

/* Custom comparator for a range request */
static int compareRange(const void* a, const void* b) {
	if ( *(AttrNumber*)a > *(AttrNumber*)b )
		return 1;
	else if ( *(AttrNumber*)a == *(AttrNumber*)b )
		return 0;
	else
		return -1;
}

/*
 * Dump the request to stderr.
 */
void ocl_dumpRequest(ocl_estimator_request_t* request) {
	unsigned int i;
	if (!request)
		return;
	fprintf(stderr, "Firing request for table: %i:\n", request->table_identifier);
	for (i=0; i<request->range_count; ++i)
		fprintf(stderr, "\tColumn %i in: [%f , %f]\n", request->ranges[i].colno, request->ranges[i].lower_bound, request->ranges[i].upper_bound);
}

int ocl_updateRequest(ocl_estimator_request_t* request,
		AttrNumber colno, float* lower_bound, float* upper_bound) {
	/*
	 * First, make sure to find the range entry for the given column.
	 * If no column exists, insert a new one.
	 */
	ocl_colrange_t* column_range = NULL;
	if (request->ranges == NULL) {
		/* We have added no ranges so far. Add this range as the first entry. */
		request->ranges = (ocl_colrange_t*)malloc(sizeof(ocl_colrange_t));
		request->range_count++;
		request->ranges->colno = colno;
		request->ranges->lower_bound = -FLT_MAX;
		request->ranges->upper_bound = FLT_MAX;
		column_range = request->ranges;
	} else {
		/* Check whether we already have a range for this column */
		column_range = bsearch(
				&colno, request->ranges, request->range_count,
				sizeof(ocl_colrange_t), &compareRange);
		if (column_range == NULL) {
			/* We have to add the column. Add storage for a new value */
			request->range_count++;
			request->ranges = (ocl_colrange_t*)realloc(request->ranges, sizeof(ocl_colrange_t)*(request->range_count));
			/* Initialize the new column range */
			column_range = &(request->ranges[request->range_count-1]);
			column_range->colno = colno;
			column_range->lower_bound = -FLT_MAX;
			column_range->upper_bound = FLT_MAX;
			/* Now we have to re-sort the array */
			qsort(request->ranges, request->range_count, sizeof(ocl_colrange_t), &compareRange);
			/* Ok, we inserted the value. Use bsearch again to get the final position of our newly inserted range. */
			column_range = bsearch(
							&colno, request->ranges, request->range_count,
							sizeof(ocl_colrange_t), &compareRange);
		}
	}
	/* Now update the found range entry with the new information */
	if (lower_bound) {
		if (column_range->lower_bound <= *lower_bound) {
			column_range->lower_bound = *lower_bound;
		}
	}
	if (upper_bound) {
		if (column_range->upper_bound >= *upper_bound) {
			column_range->upper_bound = *upper_bound;
		}
	}
	return 1;
}

int ocl_estimateSelectivity(const ocl_estimator_request_t* request,
		Selectivity* selectivity) {
	struct timeval start; gettimeofday(&start, NULL); 
	unsigned int i,j;
	// Fetch the OpenCL context, initializing it if requested.
	ocl_context_t* ctxt = ocl_getContext();
	if (ctxt == NULL)
		return 0;
	// Make sure that the registry is initialized
	if (registry == NULL)
		ocl_initializeRegistry();
	// Check the registry, whether we have an estimator for the requested table.
	ocl_estimator_t* estimator = bsearch(&(request->table_identifier),
			registry->estimators, registry->nr_of_estimators,
			sizeof(ocl_estimator_t), compareEstimator);
	if (estimator == NULL)
		return 0;
	// Check if the request can potentially be answered by the estimator 
	if (request->range_count > estimator->nr_of_dimensions) {
		return 0;
	}
	// Now prepare a request to the estimator
	float* row_ranges = (float*)malloc(2*sizeof(float)*estimator->nr_of_dimensions);
	for (i = 0; i < estimator->nr_of_dimensions; ++i) {
		row_ranges[2*i] = -FLT_MAX;
		row_ranges[2*i+1] = FLT_MAX;
	}
	j = 0;
	int found = 1;
	for (i = 0; i < request->range_count; ++i) {
		found = 0;
		for (;!found && j < estimator->nr_of_dimensions; ++j) {
			if (estimator->column_numbers[j] == request->ranges[i].colno) {
				// Make sure we adjust the request to the re-scaled data.
				row_ranges[2*j] = request->ranges[i].lower_bound / estimator->scale_factors[j];
				row_ranges[2*j+1] = request->ranges[i].upper_bound / estimator->scale_factors[j];
				found = 1;
			}
		}
		if (!found)
			break;
	}
	if (found) {
		// Prepare the requst
		clEnqueueWriteBuffer(ctxt->queue, ctxt->input_buffer, CL_FALSE, 0, 
               2*estimator->nr_of_dimensions*sizeof(float), row_ranges, 
               0, NULL, NULL);
	   *selectivity = rangeKDE(ctxt, estimator);
		// Print timing: 
		struct timeval now; gettimeofday(&now, NULL); 
		long seconds = now.tv_sec - start.tv_sec; 
		long useconds = now.tv_usec - start.tv_usec;
		long mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5; 
		fprintf(stderr, "Estimated selectivity: %f, took: %ld ms.\n", *selectivity, mtime);
		free(row_ranges);
		return 1;
	} else {
		free(row_ranges);
		return 0;
	}
}

int ocl_prepareEstimator(ocl_estimator_t* estimator,
		unsigned int n, unsigned int d, bool full_sample,
		cl_mem buffer, float* bandwidth) {
	ocl_context_t* ctxt = ocl_getContext();
	if (ctxt == NULL)
		return 0;
	estimator->data_sample = buffer;
	estimator->nr_of_dimensions = d;
	estimator->sample_size = n;
	estimator->exact = full_sample;
	// Construct the build-string
	char build_string[40];
	snprintf(build_string, 40, "-DD=%i -DSAMPLE_SIZE=%i", d, n);
	// Prepare the kernel. If we have a full sample, use the exact kernel, otherwise use KDE.
	if (full_sample) {
		fprintf(stderr, "> Using exact evaluation.\n");
		estimator->range_kernel = ocl_getKernel("exact_kde", build_string);
   } else {
		fprintf(stderr, "> Using KDE.\n");
		estimator->range_kernel = ocl_getKernel("range_kde", build_string);
	}
	// Make sure the kernel was correctly created.
	if (estimator->range_kernel == NULL) {
		return 0;
	}
	cl_int err;
	if (!full_sample) {
		// Ship the bandwidth to the kernel	
		estimator->bandwidth_buffer = clCreateBuffer(ctxt->context,
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, d*sizeof(float), bandwidth, &err);
		if (err) {
			clReleaseKernel(estimator->range_kernel);
			return 0;
		}
		err |= clSetKernelArg(estimator->range_kernel, 3, sizeof(cl_mem), &(estimator->bandwidth_buffer));
	}
	// Now prepare the arguments.
	err |= clSetKernelArg(estimator->range_kernel, 0, sizeof(cl_mem), &buffer);
	err |= clSetKernelArg(estimator->range_kernel, 1, sizeof(cl_mem), &(ctxt->result_buffer));
	err |= clSetKernelArg(estimator->range_kernel, 2, sizeof(cl_mem), &(ctxt->input_buffer));
	if (err) {
		clReleaseKernel(estimator->range_kernel);
		if (!full_sample)
			clReleaseMemObject(estimator->bandwidth_buffer);
		return 0;
	}
	// And compute the range normalization factor:
	if (full_sample)
		estimator->range_normalization_factor = 1.0f /(float)n; 
	else	
		estimator->range_normalization_factor = 0.75f /(float)n; 
	return 1;
}


unsigned int ocl_maxSampleSize(unsigned int dimensionality) {
	return (kde_samplesize*1024*1024)/(dimensionality*sizeof(float));
}

void ocl_constructEstimator(Relation rel, unsigned int rows_in_table, unsigned int dimensionality, AttrNumber* attributes, unsigned int sample_size, HeapTuple* sample) {
	unsigned int i, j;
	// Make sure we have a context
	ocl_context_t* ctxt = ocl_getContext();
	if (ctxt == NULL)
		return;
	// Make sure the registry exists.
	if (!registry)
		ocl_initializeRegistry();
	// Some Debug output
	fprintf(stderr, "Constructing an estimator for table %i.\n", rel->rd_node.relNode);
	fprintf(stderr, "\tColumns:");
	for (i=0; i<dimensionality; ++i)
		fprintf(stderr, " %i", attributes[i]);
	fprintf(stderr, "\n");extern void assign_max_kde_samplesize(int newval, void *extra);
	extern void assign_enable_kde_estimator(bool newval, void *extra);
	fprintf(stderr, "\tUsing a backing sample of %i out of %i tuples.\n", sample_size, rows_in_table);
	// Check if this estimator already exists:
	ocl_estimator_t* estimator = bsearch(&(rel->rd_node.relNode), registry->estimators, registry->nr_of_estimators, sizeof(ocl_estimator_t), compareEstimator);
	if (estimator == NULL) {
		// We have to insert a new estimator.
		registry->nr_of_estimators++;
		registry->estimators = realloc(registry->estimators, registry->nr_of_estimators*sizeof(ocl_estimator_t));
		estimator = &(registry->estimators[registry->nr_of_estimators-1]); 
		estimator->relation_id = rel->rd_node.relNode;
		qsort(registry->estimators, registry->nr_of_estimators, sizeof(ocl_estimator_t), compareEstimator); 
		estimator = bsearch(&(rel->rd_node.relNode), registry->estimators, registry->nr_of_estimators, sizeof(ocl_estimator_t), compareEstimator);
	} else {
		// Clean the estimator up
		ocl_freeEstimator(estimator);
	}
	// Update the descriptor info.
	estimator->column_numbers = attributes;
	estimator->nr_of_dimensions = dimensionality;
	estimator->sample_size = sample_size;
	/* 
    * Ok, we set up the estimator. Prepare the sample for shipping it to the device. 
    * While preparing the sample, we compute the variance for each column on the fly.
 	 * The variance is then used to normalize the data to unit variance in each dimension.
    */
	float* host_buffer = (float*)malloc(sizeof(float)*dimensionality*sample_size);
	estimator->scale_factors = (float*)malloc(sizeof(float)*dimensionality);
	for ( i = 0; i < dimensionality; ++i) {
		bool isnull;
		// Values for the online variance computation	
		unsigned int n = 0;
		float mean = 0;
		float M2 = 0;
		// Make sure we use the correct extraction function for both double and float.
		for ( j = 0; j < sample_size; ++j) {
			Oid attribute_type = rel->rd_att->attrs[attributes[i]-1]->atttypid;
			float v = 0.0f;
			if (attribute_type == FLOAT4OID)
				v = DatumGetFloat4(heap_getattr(sample[j], attributes[i], rel->rd_att, &isnull));
			else if (attribute_type == FLOAT8OID)
				v = DatumGetFloat8(heap_getattr(sample[j], attributes[i], rel->rd_att, &isnull));
			// Store in the host buffer. Adjust for access patterns on the device.
			host_buffer[j*dimensionality + i] = v;
			// And update the variance
			n++;
			float delta = v - mean;
			mean += delta / n;
			M2 += delta*(v - mean);		
		}
		estimator->scale_factors[i] = sqrt(M2/(n-1));
	}
	// Re-scale the data to unit variance
	for ( j = 0; j < sample_size; ++j) {
		for ( i = 0; i < dimensionality; ++i) {
			host_buffer[j*dimensionality + i] /= estimator->scale_factors[i];
		}
	}
	// Now compute an initial bandwidth estimate using scott's rule
	float* bandwidth = (float*)malloc(sizeof(float)*dimensionality);
	for ( i = 0; i < dimensionality; ++i) {
		bandwidth[i] = pow(sample_size, -1.0f/(float)(dimensionality+4))*sqrt(5);
	}
	// Print some debug info
	fprintf(stderr, "\tBandwidth:");
	for ( i = 0; i< dimensionality ; ++i)
		fprintf(stderr, " %f", bandwidth[i]);
	fprintf(stderr, "\n"); 
	// Push to the device.
	cl_mem device_buffer = clCreateBuffer(ctxt->context, CL_MEM_READ_ONLY, sizeof(float)*dimensionality*sample_size, NULL, NULL);
	clEnqueueWriteBuffer(ctxt->queue, device_buffer, CL_TRUE, 0, sizeof(float)*dimensionality*sample_size, host_buffer, 0, NULL, NULL);
	// And update the estimator
	ocl_prepareEstimator(estimator, sample_size, dimensionality, (rows_in_table == sample_size), device_buffer, bandwidth);
	// Clean up
	free(host_buffer);
}

void assign_ocl_use_gpu(bool newval, void *extra) {
	if (newval != ocl_use_gpu) {
		ocl_releaseRegistry();
		ocl_releaseContext();
	}
}

void assign_kde_samplesize(int newval, void *extra) {
  if (newval != kde_samplesize) {
    ocl_releaseRegistry();
    ocl_releaseContext();
  }
}

void assign_enable_kde_estimator(bool newval, void *extra) {
  if (newval != enable_kde_estimator) {
    ocl_releaseRegistry();
    ocl_releaseContext();
  }
}

bool ocl_useKDE(void) {
  return enable_kde_estimator;
}

#endif /* USE_OPENCL */
