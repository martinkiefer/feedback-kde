#ifndef KDE_EXECUTE_H
#define KDE_EXECUTE_H

#include "nodes/primnodes.h"

typedef struct PlanState PlanState;

typedef enum inclusiveness { IN, EX, EQ} inclusiveness_t;

/* Prototypes for KDE functions */
typedef struct RQClause
{
	Var	   *var;			/* The common variable of the clauses */
	inclusiveness_t   loinclusive;
	inclusiveness_t   hiinclusive;
	float8 lobound;		/* Value of a var > something clause */
	float8 hibound;		/* Value of a var < something clause */
} RQClause;

typedef struct RQClauseList
{
  struct RQClauseList *next;    /* next in linked list */
  RQClause clause;
} RQClauseList;

extern bool kde_feedback_use_collection();
extern RQClauseList *kde_get_rqlist(List *clauses);
extern int kde_finish(PlanState *node);

#endif
