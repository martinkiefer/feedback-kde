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
#include <time.h>

typedef enum bound { HIGHBOUND, LOWBOUND, EQUALITY} bound_t;


#define deparse_columns_fetch(rangetable_index, dpns) \
	((deparse_columns *) list_nth((dpns)->rtable_columns, (rangetable_index)-1))

typedef struct
{
	List	   *rtable;			/* List of RangeTblEntry nodes */
	List	   *rtable_names;	/* Parallel list of names for RTEs */
	List	   *rtable_columns; /* Parallel list of deparse_columns structs */
	List	   *ctes;			/* List of CommonTableExpr nodes */
	/* Workspace for column alias assignment: */
	bool		unique_using;	/* Are we making USING names globally unique */
	List	   *using_names;	/* List of assigned names for USING columns */
	/* Remaining fields are used only when deparsing a Plan tree: */
	PlanState  *planstate;		/* immediate parent of current expression */
	List	   *ancestors;		/* ancestors of planstate */
	PlanState  *outer_planstate;	/* outer subplan state, or NULL if none */
	PlanState  *inner_planstate;	/* inner subplan state, or NULL if none */
	List	   *outer_tlist;	/* referent for OUTER_VAR Vars */
	List	   *inner_tlist;	/* referent for INNER_VAR Vars */
	List	   *index_tlist;	/* referent for INDEX_VAR Vars */
} deparse_namespace;

typedef struct
{
	/*
	 * colnames is an array containing column aliases to use for columns that
	 * existed when the query was parsed.  Dropped columns have NULL entries.
	 * This array can be directly indexed by varattno to get a Var's name.
	 *
	 * Non-NULL entries are guaranteed unique within the RTE, *except* when
	 * this is for an unnamed JOIN RTE.  In that case we merely copy up names
	 * from the two input RTEs.
	 *
	 * During the recursive descent in set_using_names(), forcible assignment
	 * of a child RTE's column name is represented by pre-setting that element
	 * of the child's colnames array.  So at that stage, NULL entries in this
	 * array just mean that no name has been preassigned, not necessarily that
	 * the column is dropped.
	 */
	int			num_cols;		/* length of colnames[] array */
	char	  **colnames;		/* array of C strings and NULLs */

	/*
	 * new_colnames is an array containing column aliases to use for columns
	 * that would exist if the query was re-parsed against the current
	 * definitions of its base tables.	This is what to print as the column
	 * alias list for the RTE.	This array does not include dropped columns,
	 * but it will include columns added since original parsing.  Indexes in
	 * it therefore have little to do with current varattno values.  As above,
	 * entries are unique unless this is for an unnamed JOIN RTE.  (In such an
	 * RTE, we never actually print this array, but we must compute it anyway
	 * for possible use in computing column names of upper joins.) The
	 * parallel array is_new_col marks which of these columns are new since
	 * original parsing.  Entries with is_new_col false must match the
	 * non-NULL colnames entries one-for-one.
	 */
	int			num_new_cols;	/* length of new_colnames[] array */
	char	  **new_colnames;	/* array of C strings */
	bool	   *is_new_col;		/* array of bool flags */

	/* This flag tells whether we should actually print a column alias list */
	bool		printaliases;

	/*
	 * If this struct is for a JOIN RTE, we fill these fields during the
	 * set_using_names() pass to describe its relationship to its child RTEs.
	 *
	 * leftattnos and rightattnos are arrays with one entry per existing
	 * output column of the join (hence, indexable by join varattno).  For a
	 * simple reference to a column of the left child, leftattnos[i] is the
	 * child RTE's attno and rightattnos[i] is zero; and conversely for a
	 * column of the right child.  But for merged columns produced by JOIN
	 * USING/NATURAL JOIN, both leftattnos[i] and rightattnos[i] are nonzero.
	 * Also, if the column has been dropped, both are zero.
	 *
	 * If it's a JOIN USING, usingNames holds the alias names selected for the
	 * merged columns (these might be different from the original USING list,
	 * if we had to modify names to achieve uniqueness).
	 */
	int			leftrti;		/* rangetable index of left child */
	int			rightrti;		/* rangetable index of right child */
	int		   *leftattnos;		/* left-child varattnos of join cols, or 0 */
	int		   *rightattnos;	/* right-child varattnos of join cols, or 0 */
	List	   *usingNames;		/* names assigned to merged columns */
} deparse_columns;

static void kde_print_rqlist(List* rtable, PlanState  * ps, RQClause *rqlist){
	
	Bitmapset  *rels_used = NULL;
	RQClause *rqelem = NULL;
	deparse_namespace *dpns;
	List *context;
	deparse_columns *colinfo;
	List * rtable_names;
	
	rels_used = bms_add_member(rels_used, ((Scan *) ps->plan)->scanrelid);
	rtable_names = select_rtable_names_for_explain(rtable, rels_used);
	context = deparse_context_for_planstate((Node *) ps,NULL,rtable,rtable_names);
	dpns = (deparse_namespace *) list_nth(context,0);
	
	
    	while (rqlist != NULL)
	{
	    colinfo = deparse_columns_fetch(rqlist->var->varno, dpns);
	    fprintf(stderr,"%s,", colinfo->colnames[ rqlist->var->varattno - 1]);
	    fprintf(stderr, "%f,",rqlist->lobound);
	    fprintf(stderr, "%f,",rqlist->hibound);
	    rqelem = rqlist->next;
	    pfree(rqlist);
	    rqlist = rqelem;
	}
}
	
static int kde_add_rqentry(RQClause **rqlist, Var *var,double value, bound_t bound){
	RQClause *rqelem = NULL;
  	for (rqelem = *rqlist; rqelem; rqelem = rqelem->next)
	{
		if (!equal(var, rqelem->var))
			continue;
		/* Found the right group to put this clause in */
		if (bound == LOWBOUND)
		{
		    if (rqelem->lobound < value)
					rqelem->lobound = value;
		}
		else if(bound == HIGHBOUND)
		{
		    if (rqelem->hibound > value)
			rqelem->hibound = value;
		}
		else
		{
		  if(rqelem ->hibound == value || rqelem->lobound == value){
		      rqelem->hibound=value;
		      rqelem->lobound=value; 
		      return 1;
		  }
		  else
		     return 0;
		    
		}  
		if(rqelem->hibound < rqelem->lobound) 
		  return 0;
		else
		  return 1;
	}
	
	rqelem = (RQClause *) palloc(sizeof(RQClause));
	rqelem->var = var;
	if (bound == LOWBOUND)
	{
		rqelem->lobound = value;
		rqelem->hibound = get_float8_infinity();
	}
	else if (bound == HIGHBOUND)
	{
		rqelem->lobound = get_float8_infinity()*-1;
		rqelem->hibound = value;
	}
	else{
		rqelem->hibound=value;
		rqelem->lobound=value;  
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
		  		switch (get_oprrest(expr->opno))
				{
					case F_SCALARLTSEL:
						if(varonleft)
						    rc = kde_add_rqentry(&rqlist, (Var *) linitial(expr->args), DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
									    HIGHBOUND);
						else
						    rc = kde_add_rqentry(&rqlist, (Var *) lsecond(expr->args), DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
									    LOWBOUND);
						break;
					case F_SCALARGTSEL:
						if(varonleft)
						    rc = kde_add_rqentry(&rqlist, (Var *) linitial(expr->args), DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
									    LOWBOUND);
						else
						    rc = kde_add_rqentry(&rqlist, (Var *) lsecond(expr->args), DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
							 		    HIGHBOUND);
						break;
					case F_EQSEL:
						if(varonleft)
						    rc = kde_add_rqentry(&rqlist, (Var *) linitial(expr->args), DatumGetFloat8(((Const *) lsecond(expr->args))->constvalue),
									    EQUALITY);
						else
						    rc = kde_add_rqentry(&rqlist, (Var *) lsecond(expr->args), DatumGetFloat8(((Const *) linitial(expr->args))->constvalue),
									    EQUALITY);
						break;
					default:
						return NULL;
				}
				if(rc ==0 )
				  goto cleanup;
				continue;		/* drop to loop bottom */
		}
		else goto cleanup;
	}
	else goto cleanup;
    }
    return rqlist;
    
cleanup:
    //fprintf(stderr, "CLEANUP!\n");
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

	seconds = time (NULL);
	if(node == NULL)
		return 0;
   	
	if(nodeTag(node) == T_SeqScanState){
	  if(node->instrument != NULL){
	    rtable=node->instrument->kde_rtable;
	    rte = rt_fetch(((Scan *) node->plan)->scanrelid, rtable);
	    fprintf(stderr, "%ld,%s,", seconds,get_rel_name(rte->relid));
	    kde_print_rqlist(rtable, node,node->instrument->kde_rq);
	    fprintf(stderr, "%f,%f\n", 
		    (node->instrument->tuplecount + node->instrument->nfiltered2 + node->instrument->nfiltered1 + node->instrument->ntuples)/(node->instrument->nloops+1),
		    (node->instrument->tuplecount + node->instrument->ntuples)/(node->instrument->nloops+1)  
 		  );
	  }
	} 
	
	return 0;
}