#ifndef KDE_EXECUTE_H
#define KDE_EXECUTE_H

#include "nodes/primnodes.h"

typedef struct PlanState PlanState;

/* Prototypes for KDE functions */
typedef struct RQClause
{
	struct RQClause *next;		/* next in linked list */
	Var	   *var;			/* The common variable of the clauses */
	double lobound;		/* Selectivity of a var > something clause */
	double hibound;		/* Selectivity of a var < something clause */
} RQClause;

//extern int kde_init_instrumentation(PlanState *node);
//extern int kde_evaluate_instrumentation(PlanState *node, QueryDesc *queryDesc);
extern RQClause *kde_get_rqlist(List *clauses);
extern int kde_finish(PlanState *node);

#endif