#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"

//KDE
#include "postgres.h"
#include "kde_feedback/kde_feedback.h"
#include "nodes/execnodes.h"
#include "executor/instrument.h"
#include "utils/builtins.h"
#include "parser/parsetree.h"
#include "optimizer/clauses.h"
#include "optimizer/path/gpukde/ocl_estimator_api.h"
#include "catalog/pg_type.h"
#include "catalog/pg_kdefeedback.h"
#include "utils/lsyscache.h"
#include "utils/fmgroids.h"
#include "nodes/nodes.h"
#include "access/htup_details.h"
#include "utils/rel.h"
#include <time.h>
#include <float.h>

typedef enum bound { HIGHBOUND,LOWBOUND,EQUALITY} bound_t;

//OIDs from pg_operator.h
#define OID_FLOAT8_EQ 670
#define OID_FLOAT8_LT 672
#define OID_FLOAT8_LE 673
#define OID_FLOAT8_GT 674
#define OID_FLOAT8_GE 675

// GUC configuration variable.
bool kde_collect_feedback = false;

bool kde_feedback_use_collection() {
    return kde_collect_feedback;
}

// Helper function to materialize an RQlist to a buffer.
static bytea* materialize_rqlist_to_buffer(RQClauseList *rqlist) {
  // First, count how many elements are in the list.
  unsigned int elements_in_list = 0;
  RQClauseList* tmp = rqlist;
  while (tmp) {
    elements_in_list++;
    tmp = tmp->next;
  }
  if (elements_in_list == 0) return NULL;
  // Allocate the result buffer.
  bytea* buffer = palloc(sizeof(RQClause) * elements_in_list + VARHDRSZ);
  // Register the buffer size.
  SET_VARSIZE(buffer, sizeof(RQClause) * elements_in_list + VARHDRSZ);
  // And materialize all list elements.
  unsigned int pos = 0;
  char* payload = VARDATA(buffer);
  while (rqlist) {
    *((RQClause*)(&(payload[pos]))) = rqlist->clause;
    pos += sizeof(RQClause);
    rqlist = rqlist->next;
  }
  return buffer;
}

// Helper function to release the memory that ware allocated for a RQlist.
static void release_rqlist(RQClauseList* list) {
  if (list == NULL) return;
  release_rqlist(list->next);
  pfree(list);
}

static int kde_add_rqentry(RQClauseList **rqlist, Var *var, float8 value,
                           bound_t bound, inclusiveness_t inclusiveness) {
  RQClauseList *rqelem = NULL;
  for (rqelem = *rqlist; rqelem; rqelem = rqelem->next) {
    if (!equal(var, rqelem->clause.var))
      continue;
    /* Found the right group to put this clause in */

    if (bound == LOWBOUND) {
      if (rqelem->clause.lobound < value) {
        rqelem->clause.lobound = value;
        rqelem->clause.loinclusive = inclusiveness;
      } else if (rqelem->clause.lobound == value) {
        if (rqelem->clause.loinclusive == EQ && inclusiveness == EX) {
          rqelem->clause.lobound = inclusiveness;
        } else if (rqelem->clause.loinclusive == EQ && inclusiveness == EX) {
          return 0;
        } else if (rqelem->clause.loinclusive == EQ && inclusiveness == IN) {
          return 0;
        }
      }
    }
    if (bound == HIGHBOUND) {
      if (rqelem->clause.hibound > value) {
        rqelem->clause.hibound = value;
        rqelem->clause.hiinclusive = inclusiveness;
      } else if (rqelem->clause.hibound == value) {
        if (rqelem->clause.hiinclusive == IN && inclusiveness == EX) {
          rqelem->clause.hibound = inclusiveness;
        } else if (rqelem->clause.hiinclusive == EQ && inclusiveness == EX) {
          return 0;
        } else if (rqelem->clause.hiinclusive == EX && inclusiveness == IN) {
          return 0;
        }
      }
    } else if (bound == EQUALITY) {
      if (rqelem->clause.lobound == value && rqelem->clause.loinclusive == EX) {
        return 0;
      }
      if (rqelem->clause.hibound == value && rqelem->clause.hiinclusive == EX) {
        return 0;
      }
      if (rqelem->clause.lobound <= value && rqelem->clause.hibound >= value) {
        rqelem->clause.hibound = value;
        rqelem->clause.hiinclusive = inclusiveness;
        rqelem->clause.lobound = value;
        rqelem->clause.hiinclusive = inclusiveness;
        return 1;  //No consistency check necessary
      } else {
        return 0;
      }
    }
    //Consistency check
    if (rqelem->clause.hiinclusive == EX || rqelem->clause.loinclusive == EX)
      return (rqelem->clause.hibound > rqelem->clause.lobound);
    else
      return (rqelem->clause.hibound >= rqelem->clause.lobound);
  }

  rqelem = (RQClauseList *) palloc(sizeof(RQClauseList));
  rqelem->clause.var = var;

  if (bound == LOWBOUND) {
    rqelem->clause.lobound = value;
    rqelem->clause.hibound = get_float8_infinity();
    rqelem->clause.hiinclusive = IN;
    rqelem->clause.loinclusive = inclusiveness;
  } else if (bound == HIGHBOUND) {
    rqelem->clause.lobound = get_float8_infinity() * -1;
    rqelem->clause.hibound = value;
    rqelem->clause.loinclusive = IN;
    rqelem->clause.hiinclusive = inclusiveness;
  } else {
    rqelem->clause.hibound = value;
    rqelem->clause.hiinclusive = inclusiveness;
    rqelem->clause.lobound = value;
    rqelem->clause.hiinclusive = inclusiveness;
  }
  rqelem->next = *rqlist;
  *rqlist = rqelem;
  return 1;
}


RQClauseList *kde_get_rqlist(List *clauses) {
  RQClauseList *rqlist = NULL;
  RQClauseList *rqnext = NULL;

  ListCell *l;
  foreach(l, clauses)
  {
    Node *clause = (Node *) lfirst(l);
    RestrictInfo *rinfo;

    if (IsA(clause, RestrictInfo)) {
      rinfo = (RestrictInfo *) clause;
      clause = (Node *) rinfo->clause;
    } else {
      rinfo = NULL;
    }
    if (is_opclause(clause) && list_length(((OpExpr *) clause)->args) == 2) {
      OpExpr *expr = (OpExpr *) clause;
      bool varonleft = true;
      bool ok;
      int rc = 0;

      ok =
          (NumRelids(clause) == 1)
              && ((IsA(lsecond(expr->args), Const)
                  && IsA(linitial(expr->args), Var))
                  || (varonleft = false, (IsA(lsecond(expr->args), Var)
                      && IsA(linitial(expr->args), Const))));
      if (ok) {
        if (varonleft)
          ok = (((Const *) lsecond(expr->args))->consttype == FLOAT8OID);
        else
          ok = (((Const *) linitial(expr->args))->consttype == FLOAT8OID);
      }
      if (ok) {
        //We are interested in the actual operator so sadly we can't use get_oprrest here
        //Is there a more elegant way for doing this?
        switch (expr->opno) {
          case OID_FLOAT8_LE:
            if (varonleft)
              rc = kde_add_rqentry(
                  &rqlist, (Var *) linitial(expr->args),
                  DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
                  HIGHBOUND, IN);
            else
              rc = kde_add_rqentry(
                  &rqlist, (Var *) lsecond(expr->args),
                  DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
                  LOWBOUND, IN);
            break;
          case OID_FLOAT8_LT:
            if (varonleft)
              rc = kde_add_rqentry(
                  &rqlist, (Var *) linitial(expr->args),
                  DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
                  HIGHBOUND, EX);
            else
              rc = kde_add_rqentry(
                  &rqlist, (Var *) lsecond(expr->args),
                  DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
                  LOWBOUND, EX);
            break;
          case OID_FLOAT8_GE:
            if (varonleft)
              rc = kde_add_rqentry(
                  &rqlist, (Var *) linitial(expr->args),
                  DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
                  LOWBOUND, IN);
            else
              rc = kde_add_rqentry(
                  &rqlist, (Var *) lsecond(expr->args),
                  DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
                  HIGHBOUND, IN);
            break;
          case OID_FLOAT8_GT:
            if (varonleft)
              rc = kde_add_rqentry(
                  &rqlist, (Var *) linitial(expr->args),
                  DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
                  LOWBOUND, EX);
            else
              rc = kde_add_rqentry(
                  &rqlist, (Var *) lsecond(expr->args),
                  DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
                  HIGHBOUND, EX);
            break;
          case OID_FLOAT8_EQ:
            if (varonleft)
              rc = kde_add_rqentry(
                  &rqlist, (Var *) linitial(expr->args),
                  DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
                  EQUALITY, EQ);
            else
              rc = kde_add_rqentry(
                  &rqlist, (Var *) lsecond(expr->args),
                  DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
                  EQUALITY, EQ);
            break;
          default:
            goto cleanup;
        }

        if (rc == 0)
          goto cleanup;
        continue;
      } else
        goto cleanup;
    } else
      goto cleanup;
  }
  return rqlist;

cleanup:
  while (rqlist != NULL) {
    rqnext = rqlist->next;
    pfree(rqlist);
    rqlist = rqnext;
  }
  return NULL;
}

int kde_finish(PlanState *node){
	List* rtable;
  	RangeTblEntry *rte;
	
	Relation pg_database_rel;
	Datum		new_record[Natts_pg_kdefeedback];
	bool		new_record_nulls[Natts_pg_kdefeedback];
	HeapTuple	tuple;

	if(node == NULL)
		return 0;
	

   		
	if(nodeTag(node) == T_SeqScanState){
	  if(node->instrument != NULL && node->instrument->kde_rq != NULL){
	    rtable=node->instrument->kde_rtable;
	    rte = rt_fetch(((Scan *) node->plan)->scanrelid, rtable);
	    
	    
	    
	    MemSet(new_record, 0, sizeof(new_record));
	    MemSet(new_record_nulls, false, sizeof(new_record_nulls));
	    
	    bytea* rq_buffer = materialize_rqlist_to_buffer(node->instrument->kde_rq);
      release_rqlist(node->instrument->kde_rq);
	    float8 qual_tuples = (float8)(node->instrument->tuplecount + node->instrument->ntuples)/(node->instrument->nloops+1);
	    float8 all_tuples = (float8)(node->instrument->tuplecount + node->instrument->nfiltered2 + node->instrument->nfiltered1 + node->instrument->ntuples)/(node->instrument->nloops+1);
	    
	    //Hack for swallowing output when explain without analyze is called. 
	    //However, empty tables are not that interesting from a selectivity estimators point of view anyway.
	    if(qual_tuples == 0.0 && all_tuples == 0.0){
	      node->instrument->kde_rq = NULL;
	      pfree(rq_buffer);
	      return 1;
	    }
	    
	    // Notify the model maintenance of this observation.
	    ocl_notifyModelMaintenanceOfSelectivity(rte->relid, qual_tuples / all_tuples);

	    pg_database_rel = heap_open(KdeFeedbackRelationID, RowExclusiveLock);
	    
	    new_record[Anum_pg_kdefeedback_timestamp-1] = Int64GetDatum((int64)time(NULL));
	    new_record[Anum_pg_kdefeedback_relid-1] = ObjectIdGetDatum(rte->relid);
	    new_record[Anum_pg_kdefeedback_ranges-1] = PointerGetDatum(rq_buffer);
	    new_record[Anum_pg_kdefeedback_all_tuples-1] = Float8GetDatum(all_tuples);
      new_record[Anum_pg_kdefeedback_qualified_tuples-1] = Float8GetDatum(qual_tuples);
	    
	    tuple = heap_form_tuple(RelationGetDescr(pg_database_rel),
							new_record, new_record_nulls);
	    simple_heap_insert(pg_database_rel, tuple);
	
	    heap_close(pg_database_rel, RowExclusiveLock);
	    pfree(rq_buffer);
	    node->instrument->kde_rq = NULL;	    
	  }
	  
	} 
	
	return 0;
}
