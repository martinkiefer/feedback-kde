/*
 * ocl_error_metrics.h
 *
 *  Created on: 20.02.2014
 *      Author: mheimel
 */

#ifndef OCL_ERROR_METRICS_H_
#define OCL_ERROR_METRICS_H_

typedef struct error_metric {
  const char* name;
  double (*function)(double, double, double);
  double (*gradient_factor)(double, double, double);
  const char* batch_kernel_name;
} error_metric_t;

error_metric_t* ocl_getSelectedErrorMetric();

#endif /* OCL_ERROR_METRICS_H_ */
