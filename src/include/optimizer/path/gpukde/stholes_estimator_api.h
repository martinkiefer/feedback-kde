#ifndef STHOLES_H_
#define STHOLES_H_

#include "postgres_ext.h"
#include "optimizer/path/gpukde/ocl_estimator_api.h"
#include <executor/tuptable.h>
#include "access/htup_details.h"
#include "nodes/execnodes.h"


void stholes_propagateTuple(Relation rel, const TupleTableSlot* slot);

void stholes_process_feedback(PlanState *node);

void stholes_addhistogram(Oid table, AttrNumber* attributes, unsigned int dimensions);

int stholes_est(Oid rel, const ocl_estimator_request_t* request, Selectivity* selectivity);
bool stholes_enabled(void);

#endif /* STHOLES_H_ */
