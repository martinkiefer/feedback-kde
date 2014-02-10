/*
 * ocl_model_maintenance.c
 *
 *  Created on: 09.02.2014
 *      Author: mheimel
 */

#include <math.h>

#include "ocl_estimator.h"
#include "ocl_sample_maintenance.h"

// Global GUC variables
char* error_log_file_name;
FILE* error_log_file = NULL;

static void assign_error_log_file_name(const char *newval, void *extra) {
  if (error_log_file != NULL) fclose(error_log_file);
  error_log_file = fopen(newval, "w");
}

static void ocl_reportErrorToLogFile(float error) {
  fprintf(error_log_file, "%f\n", error);
  fflush(error_log_file);
}


/*
 * The error functions that we support.
 */
typedef enum error_metric {
  Absolute,    /* |target - actual| */
  Relative,    /* |target - actual| / (lambda + actual) */
  Quadratic,   /* (target - actual)^2 */
  Q            /* log(target / actual)^2 */
} error_metric_t;

error_metric_t selected_error_metric;

static double QuadraticError(float actual, float expected) {
  return (actual - expected) * (actual - expected);
}

static double QErrror(float actual, float expected) {
  double tmp = log(0.001f + actual) - log(0.001f + expected); // Constants are required to avoid computing the log of 0.
  return tmp * tmp;
}


extern void ocl_notifyModelMaintenanceOfSelectivity(
    Oid relation, RQClause* bounds, float selectivity) {
  // Check if we have an estimator for this relation.
  ocl_estimator_t* estimator = ocl_getEstimator(relation);
  if (estimator == NULL) return;
  if (!estimator->open_estimation) return;  // No registered estimation.

  // Notify the sample maintenance of this observation.
  ocl_notifySampleMaintenanceOfSelectivity(estimator, selectivity);

  // Compute the error.
  float error = 0;
  switch (selected_error_metric) {
    case Quadratic:
      error = QuadraticError(estimator->last_selectivity, selectivity);
      break;

    case Q:
      error = QErrror(estimator->last_selectivity, selectivity);
      break;
  }

  // And report it.
  ocl_reportErrorToLogFile(error);

}
