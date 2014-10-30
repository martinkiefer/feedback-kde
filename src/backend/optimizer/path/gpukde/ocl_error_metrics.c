#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"
/*
 * ocl_error_metrics.c
 *
 *  Created on: 20.02.2014
 *      Author: mheimel
 */

#include "ocl_error_metrics.h"

#include <math.h>

#include "ocl_estimator.h"

// GUC variables.
int kde_error_metric;
char* kde_estimation_quality_logfile_name;

// ############################################################
// # Define the estimation error metrics.
// ############################################################

// Functions that implement the error metrics.
static double QuadraticError(
    double estimate, double truth, double nrows) {
  return (truth - estimate) * (truth - estimate);
}
static double QuadraticErrorGradientFactor(
    double estimate, double truth, double nrows) {
  return 2 * (estimate - truth);
}
static double SquaredQErrror(double estimate, double truth, double nrows) {
  // Constants are required to avoid computing the log of 0.
  double tmp = log(1e-10 + truth) - log(1e-10 + estimate);
  return tmp * tmp;
}
static double SquaredQErrorGradientFactor(
    double estimate, double truth, double nrows) {
  return 2 * (log(1e-10 + truth) - log(1e-10 + estimate)) / (1e-10 * estimate);
}
static double AbsoluteError(double estimate, double truth, double nrows) {
  return fabs(truth - estimate);
}
static double AbsoluteErrorGradientFactor(
    double estimate, double truth, double nrows) {
  if (estimate > truth) {
    return 1;
  } else if (truth == estimate) {
    return 0;
  } else {
    return -1;
  }
}
static double RelativeError(double estimate, double truth, double nrows) {
  // Not entirely correct, but robust against zero estimates.
  return fabs(truth - estimate) / (1e-10 + truth);
}
static double RelativeErrorGradientFactor(
    double estimate, double truth, double nrows) {
  if (estimate > truth) {
    return 1.0 / (1e-10 + truth);
  } else if (truth == estimate) {
    return 0;
  } else {
    return -1.0 / (1e-10 + truth);
  }
}
static double SquaredRelativeError(
    double estimate, double truth, double nrows) {
  // Not entirely correct, but robust against zero estimates.
  double e = (truth - estimate) / (1e-10 + truth);
  return e*e;
}
static double SquaredRelativeErrorGradientFactor(
    double estimate, double truth, double nrows) {
  return 2 * (estimate - truth) / ((1e-10 + truth) * (1e-10 + truth));
}

// Array of all available metrics.
static error_metric_t error_metrics[] = {
   {
      "Absolute",  &AbsoluteError, &AbsoluteErrorGradientFactor,
      "computeBatchGradientAbsolute"
   },
   {
      "Relative", &RelativeError, &RelativeErrorGradientFactor,
      "computeBatchGradientRelative"
   },
   {
      "Quadratic", &QuadraticError, &QuadraticErrorGradientFactor,
      "computeBatchGradientQuadratic"
   },
   {
      "SquaredQError", &SquaredQErrror, &SquaredQErrorGradientFactor,
      "computeBatchGradientQ"
   },
   {
      "SquaredRelative", &SquaredRelativeError,
      &SquaredRelativeErrorGradientFactor,
      "computeBatchGradientSquaredRelative"
   }
};

error_metric_t* ocl_getSelectedErrorMetric() {
  return &(error_metrics[kde_error_metric]);
}

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
  fprintf(estimation_quality_log_file, "Relation ID");
  for (i=0; i<sizeof(error_metrics)/sizeof(error_metric_t); ++i) {
    fprintf(estimation_quality_log_file, " ; %s", error_metrics[i].name);
  }
  fprintf(estimation_quality_log_file, " ; Estimate ; Truth ; Tuples ");
  fprintf(estimation_quality_log_file, "\n");
  fflush(estimation_quality_log_file);
}

bool ocl_reportErrors(void) {
  return estimation_quality_log_file != NULL;
}

void ocl_reportErrorToLogFile(
    Oid relation, double estimate, double truth, double nrows) {
  if (estimation_quality_log_file == NULL) return;
  // Compute the estimation error for all metrics and write them to the file.
  unsigned int i;
  fprintf(estimation_quality_log_file, "%u", relation);
  for (i=0; i<sizeof(error_metrics)/sizeof(error_metric_t); ++i) {
    double error = (*(error_metrics[i].function))(estimate, truth, nrows);
    fprintf(estimation_quality_log_file, " ; %.8f", error);
   }
   fprintf(estimation_quality_log_file, " ; %e ; %e", estimate, truth);
   fprintf(estimation_quality_log_file, " ; %lu", (unsigned long ) nrows);
   fprintf(estimation_quality_log_file, "\n");
   fflush(estimation_quality_log_file);
}

