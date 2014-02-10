#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
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
char* kde_estimation_quality_logfile_name;

// ############################################################
// # Define estimation error metrics.
// ############################################################

typedef struct error_metric {
	const char* name;
	float (*function)(float,float);
} error_metric_t;


// Functions that implement the error metrics.
static float QuadraticError(float actual, float expected) {
  return (actual - expected) * (actual - expected);
}
static float QErrror(float actual, float expected) {
  // Constants are required to avoid computing the log of 0.
  float tmp = log(0.001f + actual) - log(0.001f + expected);
  return tmp * tmp;
}
static float AbsoluteError(float actual, float expected) {
  return fabs(actual - expected);
}
static float RelativeError(float actual, float expected) {
  // Not entirely correct, but robust against zero estimates.
  return fabs(actual - expected) / (0.001f + expected);
}

// Array of all available metrics.
static error_metric_t error_metrics[] = {
   {
      "Absolute",  &AbsoluteError
   },
   {
      "Relative", &RelativeError
   },
   {
      "Quadratic", &QuadraticError
   },
   {
      "Q", &QErrror
   }
};

// ############################################################
// # Code for estimation error reporting.
// ############################################################

static FILE* estimation_quality_log_file = NULL;

void assign_kde_estimation_quality_logfile_name(const char *newval, void *extra) {
  if (estimation_quality_log_file != NULL) fclose(estimation_quality_log_file);
  estimation_quality_log_file = fopen(newval, "w");
  if (estimation_quality_log_file == NULL) return;
  // Write a header to the file to specify all registered error metrics.
  unsigned int i;
  for (i=0; i<sizeof(error_metrics)/sizeof(error_metric_t); ++i) {
	  if (i>0) fprintf(estimation_quality_log_file, " ; ");
	  fprintf(estimation_quality_log_file, "%s", error_metrics[i].name);
  }
  fprintf(estimation_quality_log_file, "\n");
  fflush(estimation_quality_log_file);
}

static void ocl_reportErrorToLogFile(float actual, float expected) {
  if (estimation_quality_log_file == NULL) return;
  // Compute the estimation error for all metrics and write them to the file.
  unsigned int i;
  for (i=0; i<sizeof(error_metrics)/sizeof(error_metric_t); ++i) {
 	  if (i>0) fprintf(estimation_quality_log_file, " ; ");
 	  float error = (*(error_metrics[i].function))(actual, expected);
 	  fprintf(estimation_quality_log_file, "%.3f", error);
   }
   fprintf(estimation_quality_log_file, "\n");
   fflush(estimation_quality_log_file);
}

extern void ocl_notifyModelMaintenanceOfSelectivity(
    Oid relation, RQClause* bounds, float selectivity) {
  // Check if we have an estimator for this relation.
  ocl_estimator_t* estimator = ocl_getEstimator(relation);
  if (estimator == NULL) return;
  if (!estimator->open_estimation) return;  // No registered estimation.

  // Notify the sample maintenance of this observation.
  ocl_notifySampleMaintenanceOfSelectivity(estimator, selectivity);

  // Write the error to the log file.
  ocl_reportErrorToLogFile(estimator->last_selectivity, selectivity);

  // We are done.
  estimator->open_estimation = false;
}
