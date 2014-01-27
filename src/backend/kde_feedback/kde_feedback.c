//KDE
#include "postgres.h"
#include "kde_feedback/kde_feedback.h"
#include "nodes/execnodes.h"
#include "executor/instrument.h"
#include "utils/builtins.h"
#include "parser/parsetree.h"
#include "optimizer/clauses.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "utils/fmgroids.h"
#include "nodes/nodes.h"
#include "access/htup_details.h"
#include "utils/rel.h"
#include <time.h>
#include <float.h>

typedef enum bound { HIGHBOUND,LOWBOUND,EQUALITY} bound_t;

#define KdeRelationId 3779
#define kde_num_rels 5
#define Anum_kde_time 1
#define Anum_kde_relid 2
#define Anum_kde_ranges 3
#define Anum_kde_all_tuples 4
#define Anum_kde_qualified_tuples 5

//OIDs from pg_operator.h
#define OID_FLOAT8_EQ 670
#define OID_FLOAT8_LT 672
#define OID_FLOAT8_LE 673
#define OID_FLOAT8_GT 674
#define OID_FLOAT8_GE 675

bool enable_kde_feedback_collection = false;

bool kde_feedback_use_collection(){
    return enable_kde_feedback_collection;
}

static char *kde_print_rqlist(RQClause *rqlist){
	RQClause *rqelem;
	int size = 200;
	char *ranges = palloc(size*sizeof(char)+1);
	unsigned long chars = 0;
	int left = size;

    	while (rqlist != NULL)
	{
	    int o_size;
	    if(rqlist->next != NULL)
		o_size = snprintf(ranges+chars,left,"%d,%.*g,%.*g,", (int) rqlist->var->varattno,DBL_DIG,rqlist->lobound,DBL_DIG,rqlist->hibound);
	    else
		o_size = snprintf(ranges+chars,left,"%d,%.*g,%.*g", (int) rqlist->var->varattno,DBL_DIG,rqlist->lobound,DBL_DIG,rqlist->hibound);
	    
	    if(o_size <= left){
	      chars += o_size;
	      left -= o_size;
	    }
	    else{ //Our buffer was not large enough.
	      size += 200;
	      ranges = repalloc(ranges, size*sizeof(char)+1);
	      left += 200;
	      continue;
	    }
	    rqelem = rqlist->next;
	    pfree(rqlist);
	    rqlist = rqelem;
	    
	}
	return ranges;
}
	
static int kde_add_rqentry(RQClause **rqlist, Var *var,float8 value, bound_t bound, inclusiveness_t inclusiveness){
	RQClause *rqelem = NULL;
  	for (rqelem = *rqlist; rqelem; rqelem = rqelem->next)
	{
		if (!equal(var, rqelem->var))
			continue;
		/* Found the right group to put this clause in */
		
		if (bound == LOWBOUND)
		{
		    if(rqelem->lobound < value){
			rqelem->lobound = value;
			rqelem->loinclusive = inclusiveness;
		    }
		    else if(rqelem->lobound == value){
			if(rqelem->loinclusive == EQ && inclusiveness == EX ){
			    rqelem->lobound=inclusiveness;
			}
			else if(rqelem->loinclusive == EQ && inclusiveness == EX){
			    return 0;
			}
			else if(rqelem->loinclusive == EQ && inclusiveness == IN){
			    return 0;
			}
		    }
		}
		if (bound == HIGHBOUND)
		{
		    if(rqelem->hibound > value){
			rqelem->hibound = value;
			rqelem->hiinclusive = inclusiveness;
		    }
		    else if(rqelem->hibound == value){
			if(rqelem->hiinclusive == IN && inclusiveness == EX ){
			    rqelem->hibound=inclusiveness;
			}
			else if(rqelem->hiinclusive == EQ && inclusiveness == EX){
			    return 0;
			}
			else if(rqelem->hiinclusive == EX && inclusiveness == IN){
			    return 0;
			}
		    }
		}
		else if(bound == EQUALITY){
		    if(rqelem->lobound == value && rqelem->loinclusive == EX){
			  return 0;
		    }
		    if(rqelem->hibound == value && rqelem->hiinclusive == EX){
			  return 0;
		    }
		    if(rqelem->lobound <= value && rqelem->hibound >= value){
		      	rqelem->hibound=value;
			rqelem->hiinclusive = inclusiveness;
			rqelem->lobound=value;  
			rqelem->hiinclusive = inclusiveness;
			return 1; //No consistency check necessary
		    }
		    else {
		       return 0;
		    } 
		}
		//Consistency check
		if(rqelem->hiinclusive == EX || rqelem->loinclusive == EX)
		  return (rqelem->hibound > rqelem->lobound);
		else
		  return (rqelem->hibound >= rqelem->lobound);
	}
	
	rqelem = (RQClause *) palloc(sizeof(RQClause));
	rqelem->var = var;
	
	if (bound == LOWBOUND)
	{
		rqelem->lobound = value;
		rqelem->hibound = get_float8_infinity();
		rqelem->hiinclusive = IN;
		
		rqelem->loinclusive = inclusiveness;
	}
	else if (bound == HIGHBOUND)
	{
		rqelem->lobound = get_float8_infinity()*-1;
		rqelem->hibound = value;
		rqelem->loinclusive = IN;
		
		rqelem->hiinclusive = inclusiveness;
	}
	else{
		rqelem->hibound=value;
		rqelem->hiinclusive = inclusiveness;
		rqelem->lobound=value;  
		rqelem->hiinclusive = inclusiveness;
	}
	rqelem->next = *rqlist;
	*rqlist = rqelem;
	return 1;
}


RQClause *kde_get_rqlist(List *clauses){
    RQClause *rqlist = NULL;
    RQClause *rqnext = NULL;
    
    ListCell   *l;
    foreach(l, clauses)
    {
      	Node	   *clause = (Node *) lfirst(l);
	RestrictInfo *rinfo;
	
	if (IsA(clause, RestrictInfo))
	{
		rinfo = (RestrictInfo *) clause;
		clause = (Node *) rinfo->clause;
	}
	else 
	{
		rinfo = NULL;
	}
	if (is_opclause(clause) && list_length(((OpExpr *) clause)->args) == 2)
	{
	  	OpExpr	   *expr = (OpExpr *) clause;
		bool		varonleft = true;
		bool		ok;
		int rc = 0;
				
		ok = (NumRelids(clause) == 1) &&
			((IsA(lsecond(expr->args), Const) && IsA(linitial(expr->args), Var) ) ||
			(varonleft = false,
			(IsA(lsecond(expr->args), Var) && IsA(linitial(expr->args), Const))));
		if(ok){
		  if(varonleft)
		    ok = (((Const *) lsecond(expr->args))->consttype == FLOAT8OID); 
		  else
		    ok = (((Const *) linitial(expr->args))->consttype == FLOAT8OID); 
		}
		if(ok){
				//We are interested in the actual operator so sadly we can't use get_oprrest here
				//Is there a more elegant way for doing this?
		  		switch (expr->opno)
				{
					case OID_FLOAT8_LE:
						if(varonleft)
						    rc = kde_add_rqentry(&rqlist, (Var *) linitial(expr->args), DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
									    HIGHBOUND,IN);
						else
						    rc = kde_add_rqentry(&rqlist, (Var *) lsecond(expr->args), DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
									    LOWBOUND,IN);
						break;
					case OID_FLOAT8_LT:
						if(varonleft)
						    rc = kde_add_rqentry(&rqlist, (Var *) linitial(expr->args), DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
									    HIGHBOUND,EX);
						else
						    rc = kde_add_rqentry(&rqlist, (Var *) lsecond(expr->args), DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
									    LOWBOUND,EX);
						break;
					case OID_FLOAT8_GE:
						if(varonleft)
						    rc = kde_add_rqentry(&rqlist, (Var *) linitial(expr->args), DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
									    LOWBOUND,IN);
						else
						    rc = kde_add_rqentry(&rqlist, (Var *) lsecond(expr->args), DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
							 		    HIGHBOUND,IN);
						break;
					case OID_FLOAT8_GT:
						if(varonleft)
						    rc = kde_add_rqentry(&rqlist, (Var *) linitial(expr->args), DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
									    LOWBOUND,EX);
						else
						    rc = kde_add_rqentry(&rqlist, (Var *) lsecond(expr->args), DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
							 		    HIGHBOUND,EX);
						break;
					case OID_FLOAT8_EQ:
						if(varonleft)
						    rc = kde_add_rqentry(&rqlist, (Var *) linitial(expr->args), DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),EQUALITY,EQ);
						else
						    rc = kde_add_rqentry(&rqlist, (Var *) lsecond(expr->args), DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),EQUALITY,EQ);
						break;
					default:
						goto cleanup;
				}

				if(rc == 0 )
				  goto cleanup;
				continue;		
		}
		else goto cleanup;
	}
	else goto cleanup;
    }
    return rqlist;
    
cleanup:
    while(rqlist != NULL){
	rqnext = rqlist->next;
	pfree(rqlist);
	rqlist=rqnext;
    }
    return NULL;
}

int kde_finish(PlanState *node){
	List* rtable;
  	RangeTblEntry *rte;
	time_t seconds;
	
	Relation pg_database_rel;
	Datum		new_record[kde_num_rels];
	bool		new_record_nulls[kde_num_rels];
	HeapTuple	tuple;

	if(node == NULL)
		return 0;
	

   		
	if(nodeTag(node) == T_SeqScanState){
	  if(node->instrument != NULL && node->instrument->kde_rq != NULL){
	    rtable=node->instrument->kde_rtable;
	    rte = rt_fetch(((Scan *) node->plan)->scanrelid, rtable);
	    
	    
	    
	    MemSet(new_record, 0, sizeof(new_record));
	    MemSet(new_record_nulls, false, sizeof(new_record_nulls));
	    
	    char * rq_string = kde_print_rqlist(node->instrument->kde_rq);
	    float8 qual_tuples = (float8)(node->instrument->tuplecount + node->instrument->ntuples)/(node->instrument->nloops+1);
	    float8 all_tuples = (float8)(node->instrument->tuplecount + node->instrument->nfiltered2 + node->instrument->nfiltered1 + node->instrument->ntuples)/(node->instrument->nloops+1);
	    
	    //Hack for swallowing output when explain without analyze is called. 
	    //However, empty tables are not that interesting from a selectivity estimators point of view anyway.
	    if(qual_tuples == 0.0 && all_tuples == 0.0){
		node->instrument->kde_rq = NULL;
		pfree(rq_string);
		return 1;
	    }
	    
	    pg_database_rel = heap_open(KdeRelationId, RowExclusiveLock);
	    
	    seconds = time (NULL);
	    new_record[Anum_kde_time-1] = Int64GetDatum((int64) seconds);
	    new_record[Anum_kde_relid-1] = ObjectIdGetDatum(rte->relid);
	    new_record[Anum_kde_ranges-1] = CStringGetTextDatum(rq_string);
	    new_record[Anum_kde_qualified_tuples-1] = Float8GetDatum(qual_tuples);
	    new_record[Anum_kde_all_tuples-1] = Float8GetDatum(all_tuples);
	    
	    tuple = heap_form_tuple(RelationGetDescr(pg_database_rel),
							new_record, new_record_nulls);
	    simple_heap_insert(pg_database_rel, tuple);
	
	    heap_close(pg_database_rel, RowExclusiveLock);
	    pfree(rq_string);
	    node->instrument->kde_rq = NULL;	    
	  }
	  
	} 
	
	return 0;
}