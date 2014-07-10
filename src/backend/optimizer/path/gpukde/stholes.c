#include "optimizer/path/gpukde/stholes_estimator_api.h"
#include "ocl_estimator.h"
#include <executor/tuptable.h>
#include <math.h>
#include <float.h>
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "nodes/plannodes.h"
#include <nodes/execnodes.h>
#include "executor/instrument.h"
#include "executor/tuptable.h"


typedef struct st_hole st_hole;

/**
 * An sthole instance. 
 * Contains the number of tuples it contains, its bounds and its children
 */ 
typedef struct st_hole {
  kde_float_t tuples;
  
  int nr_children;
  st_hole* children;
  
  kde_float_t* bounds; //Bounds of this hole
  
  kde_float_t counter; //Working counter for the statistics step
} st_hole_t;


/**
 * The head of an stholes histogram. 
 * Contains additional meta information
 */ 
 typedef struct st_head {
  st_hole_t hole;
  
  int holes;
  int max_holes;
  //Meta information
  unsigned int dimensions;
  Oid table;
  int32 columns;
  unsigned int* column_order;
  st_hole_t last_query;
  kde_float_t epsilon;
  kde_float_t last_selectivity;
  
  int process_feedback;
} st_head_t;

st_head_t* current = NULL;
bool stholes_enable;
int stholes_hole_limit;

/** 
 * Check if stholes is stholes_enabled in guc
 */
bool stholes_enabled(){
  return stholes_enable;
}  

static void _printTree(st_head_t* head, st_hole_t* hole, int depth);



/**
 * Create a new empty st hole
 */
static st_hole_t createNewSthole(const st_head_t* head){
  int i = 0;
  st_hole_t hole;
  
  hole.bounds = (kde_float_t*) malloc(sizeof(kde_float_t)*head->dimensions*2);
  hole.tuples = 0.0;
  hole.counter = 0.0;
  hole.nr_children = 0;
  hole.children = NULL;
    //Initialize bound with +/- infinfity
  for(i=0 ; i<head->dimensions ; i++){
    hole.bounds[i*2] = INFINITY;
    hole.bounds[i*2+1] = -INFINITY;
  }
  
  return hole;
}

/**
 * Create a new head bucket for the histogram
 */
static st_head_t* createNewHistogram(Oid table,AttrNumber* attributes, unsigned int dimensions){
  int i = 0;
  st_head_t* head = (st_head_t*) calloc(1,sizeof(st_head_t));
  
  head->dimensions = dimensions;
  head->table = table;
  head->process_feedback = 0;
  head->max_holes = stholes_hole_limit;
  head->holes = 1;
  head->column_order = calloc(1, 32 * sizeof(unsigned int));
  
  head->last_query = createNewSthole(head);
  head->hole = createNewSthole(head);
  
  if(sizeof(kde_float_t) == sizeof(double)){
    head->epsilon = DBL_EPSILON;
  }
  else {
    head->epsilon = FLT_EPSILON;
  }  
  
  for (i = 0; i<dimensions; ++i) {
     head->columns |= 0x1 << attributes[i];
     head->column_order[attributes[i]] = i;
  }
  
  //Initialize bound with +/- infinfity
  for(i=0 ; i<dimensions ; i++){
    head->hole.bounds[i*2] = INFINITY;
    head->hole.bounds[i*2+1] = -INFINITY;
  }
  
  return head;
}

/**
 * Release the ressources of an st hole
 */
static void releaseRessources(st_hole_t* hole){
  free(hole->bounds);
  free(hole->children);
}  

static void _destroyHistogram(st_hole_t* hole){
  int i = 0;
  for(i = 0; i < hole->nr_children; i++){
    _destroyHistogram(hole->children+i);
  }
  releaseRessources(hole);
}  

static void destroyHistogram(st_head_t* head){
  _destroyHistogram((st_hole_t*) head);
  free(head);
}  


//Convert the last query to an sthole.
//We can then simply use our standard functions for vBox and v for it.
static void setLastQuery(const st_head_t* head, const ocl_estimator_request_t* request){
  int i = 0;
  for(i = 0; i < request->range_count; i++){
    //Add tiny little epsilons, if necessary, to account for the [) kind of buckets
    if(request->ranges[i].lower_included)
      head->last_query.bounds[head->column_order[request->ranges[i].colno]*2] = request->ranges[i].lower_bound;
    else
      head->last_query.bounds[head->column_order[request->ranges[i].colno]*2] = request->ranges[i].lower_bound + abs(request->ranges[i].lower_bound) * head->epsilon;
    
    if(request->ranges[i].upper_included)
      head->last_query.bounds[head->column_order[request->ranges[i].colno]*2+1] = request->ranges[i].upper_bound + abs(request->ranges[i].upper_bound) * head->epsilon;
    else
      head->last_query.bounds[head->column_order[request->ranges[i].colno]*2+1] = request->ranges[i].upper_bound;
  }     
}  


/**
 * Introduce new child to parent
 */
static void registerChild(const st_head_t* head, st_hole_t* parent, const st_hole_t* child){
    parent->nr_children++;
    parent->children = realloc(parent->children,parent->nr_children * sizeof(st_hole_t));
    parent->children[parent->nr_children-1] = *child;
}

/**
 * Remove the child at position pos in bucket parent
 */
static void unregisterChild(st_head_t* head, st_hole_t* parent, int pos){
    Assert(pos >= 0);
    Assert(pos < parent->nr_children);
    
    if(pos != parent->nr_children-1){
	parent->children[pos]=parent->children[parent->nr_children-1];
    }
    parent->nr_children--;
    parent->children = realloc(parent->children,parent->nr_children * sizeof(st_hole_t));
}

/**
 * vBox operator from the paper
 * bucket volume (children included)
 */
static kde_float_t vBox(const st_head_t* head, const st_hole_t* hole){
  kde_float_t volume = 1.0;
  int i = 0;
  for(i=0; i < head->dimensions; i++){
      volume *= hole->bounds[i*2+1] - hole->bounds[i*2];
  }  
  if(volume < 0) return 0;
  return volume;
}

/**
 * v function from the paper
 * bucket volume (children excluded)
 */
static kde_float_t v(const st_head_t* head, const st_hole_t* hole){
  kde_float_t v = vBox(head,hole);
  int i = 0;
  for(i=0; i < hole->nr_children; i++){
      v -= vBox(head,hole->children + i);
  }  
  Assert(v >= -1.0); //This usually means that something went terribly wrong.
  if(v < 0) return 0;
  return v;
}

/**
 * Calculate the intersection bucket with the last query (head->lastquery).
 * Stores it in target_hole
 */
static void intersection_with_last_query(const st_head_t* head, const st_hole_t* hole, st_hole_t* target_hole){
  int i = 0;
  for(i = 0; i < head->dimensions; i++){
    target_hole->bounds[2*i] = fmax(head->last_query.bounds[2*i],hole->bounds[2*i]);
    target_hole->bounds[2*i+1] = fmin(head->last_query.bounds[2*i+1],hole->bounds[2*i+1]);
  }  
}

/**
 * Calculate the smallest box containing hole and target_hole.
 * Stores the result in target_hole
 */
static void boundingBox(const st_head_t* head, const st_hole_t* hole, st_hole_t* target_hole){
  int i = 0;
  for(i = 0; i < head->dimensions; i++){
    target_hole->bounds[2*i] = fmin(target_hole->bounds[2*i],hole->bounds[2*i]);
    target_hole->bounds[2*i+1] = fmax(target_hole->bounds[2*i+1],hole->bounds[2*i+1]);
  }  
}  
typedef enum {FULL12, FULL21, NONE, PARTIAL, EQUALITY} intersection_t;

/* Calculates the strongest relationship between two histogram buckets
 * EQUALITY: 	hole1 and two are the same
 * FULL12: 	hole1 fully contains hole2
 * FULL21: 	hole2 fully contains hole1
 * PARTIAL:	The holes have a partial intersection
 * NONE:	The holes are disjunct
 */
static intersection_t getIntersectionType(const st_head_t* head, const st_hole_t* hole1, const st_hole_t* hole2){
  int i = 0;
  int enclosed12 = 1;
  int enclosed21 = 1;
  for(i = 0; i < head->dimensions; i++){
    //Case 1: We have no intersection with this hole
    //If this does not intersect with one of the intervals of the box, we have nothing to do.
    //Neither have our children.
    if(hole1->bounds[2*i+1] <= hole2->bounds[2*i] || hole1->bounds[2*i] >= hole2->bounds[2*i+1]){
      return NONE;
    }
    
    if(! (hole2->bounds[2*i] >= hole1->bounds[2*i] && hole2->bounds[2*i+1] <= hole1->bounds[2*i+1])){
      enclosed12 = 0;
    }
    
    if(! (hole1->bounds[2*i] >= hole2->bounds[2*i] && hole1->bounds[2*i+1] <= hole2->bounds[2*i+1])){
      enclosed21 = 0;
    } 
  }
  if(enclosed21 && enclosed12){
    return EQUALITY;
  }
  else if(enclosed21){
    return FULL21;
  }
  else if(enclosed12){
    return FULL12;
  } 
  else {
    return PARTIAL;
  }  
}

/** 
 * Debugging function, can be used to check the histogram for inconsistencies regarding the disjunctiveness of buckets
 */
static int _disjunctivenessTest(st_head_t* head, st_hole_t* hole){
  int i,j = 0;
  for(i = 0; i < hole->nr_children; i++){
    for(j = i+1; j < hole->nr_children; j++){
      if(getIntersectionType(head,hole->children+i,hole->children+j) != NONE){
	fprintf(stderr, "Intersection between %i %i is %i\n",i,j,getIntersectionType(head,hole->children+i,hole->children+j));
	return 0;
      }
    }  
  }
  return 1;
}  

/**
 * Aggregate estimated tuples recursively
 */
static kde_float_t _est(const st_head_t* head, const st_hole_t* hole, kde_float_t* intersection_vol){
  int i = 0;
  kde_float_t est = 0.0;
  kde_float_t v_q_i_b = 0.0;
  st_hole_t q_i_b;
  kde_float_t vh;
  
  *intersection_vol = 0;
  if(getIntersectionType(head,hole,&head->last_query) == NONE) return est;
  
  //Calculate v(q b)
  q_i_b = createNewSthole(head);

  intersection_with_last_query(head,hole,&q_i_b);
  
  *intersection_vol = vBox(head,&q_i_b);
  
  v_q_i_b = *intersection_vol;
  
  //Do we have business in this hole?
  
  for(i = 0; i < hole->nr_children; i++){
    kde_float_t child_intersection;
    est += _est(head, hole->children + i,&child_intersection);
    v_q_i_b -=  child_intersection;
  }
  if(v_q_i_b <= 0.0){
    v_q_i_b = 0.0;
  }
  //This is a corner case, we have to fetch.
  vh =  v(head,hole);
  
  //This can occur when we are unlucky.
  if(vh >= abs(head->epsilon*vBox(head,hole))) 
    est += hole->tuples * (v_q_i_b / vh); 
  
  
  //fprintf(stderr, "Est: %f * %f / %f\n",hole->tuples, v_q_i_b,vh);
  releaseRessources(&q_i_b);
  
  return est;
}  


static int _propagateTuple(st_head_t* head,st_hole_t* hole, kde_float_t* tuple){
  int i = 0;
  //Let see, if this point is within our bounds
  for(i = 0; i < head->dimensions; i++){
      //If it is not, our father will here about this
      if(tuple[i] >= hole->bounds[2*i+1] || tuple[i] < hole->bounds[2*i]) return 0;
  }
  
  //Tell the children about the point
  for(i = 0; i < hole->nr_children; i++){
    //If one of our children claims this point, we tell our parents about it
    if(_propagateTuple(head,hole->children+i,tuple) == 1) return 1;
  }
  
  //None of the children showed interest in the point
  hole->counter++;
  return 1;
}

/**
 * Recursively reset the counter for all holes 
 */
static void resetAllCounters(st_hole_t* hole){
  int i = 0;
  hole->counter = 0;
  for(i = 0; i < hole->nr_children; i++){
    //If one of our children claims this point, we tell our parents about it
    resetAllCounters(hole->children+i);
  }
}



//Get the volume of intersection when shrinking it along dimension such that is does not intersect with hole
static kde_float_t getReducedVolume(st_head_t* head, st_hole_t* intersection, st_hole_t* hole, int dimension){
  
  kde_float_t vol = vBox(head,intersection);
  
  //Remove the selected dimension
  vol /= intersection->bounds[dimension*2+1] - intersection->bounds[dimension*2];
  

  if(intersection->bounds[dimension*2] >= hole->bounds[dimension*2] && intersection->bounds[dimension*2] < hole->bounds[dimension*2+1] ){
    //If the dimension is completely located inside the box, the dimension is not eligible for dimension redecution
    if(intersection->bounds[dimension*2+1] <= hole->bounds[dimension*2+1]){
	return -INFINITY;
    } 
    
    //If the lower bound is located inside the other box, we have to exchange it.
    return vol * (intersection->bounds[dimension*2+1] - hole->bounds[dimension*2+1]);
  }
  else if(intersection->bounds[dimension*2+1] > hole->bounds[dimension*2] && intersection->bounds[dimension*2+1] <= hole->bounds[dimension*2+1]){
    return vol * (hole->bounds[dimension*2] - intersection->bounds[dimension*2]);
  }
  //The hole intervall is completely in the intersection. In this case, we have a choice
  else if(hole->bounds[dimension*2] >= intersection->bounds[dimension*2] && hole->bounds[dimension*2+1] <= intersection->bounds[dimension*2+1]){
    return fmax(vol * (hole->bounds[dimension*2] - intersection->bounds[dimension*2]),
		vol * (intersection->bounds[dimension*2+1] - hole->bounds[dimension*2+1]));
  }
  else {
    //If the dimensions are completely distinct, we should not have called this method.
    //_printTree(head,intersection,0);
    //_printTree(head,hole,0);
    Assert(0);
    return 0.0;
  }
}

static void shrink(st_head_t* head, st_hole_t* intersection, st_hole_t* hole, int dimension){
  if(intersection->bounds[dimension*2] >= hole->bounds[dimension*2] && intersection->bounds[dimension*2] < hole->bounds[dimension*2+1] ){
    //If the dimension is completely located inside the box, the dimension is not eligible for dimension redecution
    if(intersection->bounds[dimension*2+1] <= hole->bounds[dimension*2+1]){
	Assert(0); //Not here.
    } 
    
    //If the lower bound is located inside the other box, we have to exchange it.
    intersection->bounds[dimension*2] = hole->bounds[dimension*2+1];
  }
  else if(intersection->bounds[dimension*2+1] > hole->bounds[dimension*2] && intersection->bounds[dimension*2+1] <= hole->bounds[dimension*2+1]){
    intersection->bounds[dimension*2+1] = hole->bounds[dimension*2];
  }
  else if(hole->bounds[dimension*2] >= intersection->bounds[dimension*2] && hole->bounds[dimension*2+1] <= intersection->bounds[dimension*2+1]){
    if( hole->bounds[dimension*2] - intersection->bounds[dimension*2] > intersection->bounds[dimension*2+1] - hole->bounds[dimension*2+1]){
      intersection->bounds[dimension*2+1] = hole->bounds[dimension*2];
    }
    else {
      intersection->bounds[dimension*2] = hole->bounds[dimension*2+1];
    }  
  }
  else {
    //If not at least one of the coordinates is located inside the box, we should not have called this method.
    Assert(0);
  }
}  

//Returns the dimension that offers the least reduced volume when progressively shrinking
//intersection such that it does not partially intersect with hole anymore.
static unsigned int minReducedVolumeDimension(st_head_t* head, st_hole_t* intersection, st_hole_t* hole){
  int i = 0;
  int max_dim = -1;
  kde_float_t max_vol = -INFINITY;
	    
  for(i = 0; i < head->dimensions; i++){
    kde_float_t vol = getReducedVolume(head, intersection, hole, i);
    if(vol > max_vol){
      max_vol = vol;
      max_dim = i;
    }
  }
  
  Assert(max_vol != -INFINITY && max_dim != -1);
  return max_dim;
}  

/**
 * Finds candidate holes and drills them
 */
static void _drillHoles(st_head_t* head, st_hole_t* parent, st_hole_t* hole){
  st_hole_t candidate,tmp;

  int i, pos, old_hole_size = 0;
  intersection_t type;
  kde_float_t v_qib;
  kde_float_t parent_vol;
  
  if(getIntersectionType(head,&(head->last_query),hole) == NONE) return;
  if(parent != NULL) pos = hole - parent->children; //Remeber the position of the hole before we mess with the array
  candidate= createNewSthole(head);
  tmp = createNewSthole(head);
  //Get the intersection with the last query for this hole.
  intersection_with_last_query(head,hole,&candidate);
  v_qib = vBox(head,&candidate); //*Will be adjusted to the correct value later
  //fprintf(stderr,"Candidate hole:\n");
  //_printTree(head,&candidate,0);
  type = getIntersectionType(head,hole,&candidate);
  
  //Shrinking
  switch(type){
    case NONE:
      return;
    
    //Case 2: We have complete intersection with this hole. Update stats.
    case EQUALITY:
      hole->tuples = hole->counter;
      goto nohole;     
      
    
    case FULL12:
      //We will use the tmp
      for(i = 0; i < hole->nr_children; i++){
	if(getIntersectionType(head,&(head->last_query),hole->children+i) != NONE){
	  intersection_with_last_query(head,hole->children+i,&tmp);
	  v_qib -= vBox(head,&tmp);
	}  
      } 
      releaseRessources(&tmp);

      for(;;){ //We have to do this until all children are either none intersecting or included
	int changed = 0;
	for(i = 0; i < hole->nr_children; i++){
	  //Full intersections
	  intersection_t type = getIntersectionType(head,hole->children+i,&candidate);
	  //Full and no intersection are no problem
	  if(type == FULL12){
	    goto nohole; 
	  }  
	  else if(type == NONE){
	    continue;
	  }
	  else if(type == EQUALITY){
	    //The child can handle this case on its own.
	    goto nohole;
	  }
	  else if(type == FULL21){
	    continue; //The child is completely located inside the intersection.
	  }  
	  else {
	    unsigned int max_dim = minReducedVolumeDimension(head,&candidate,hole->children+i);
	    shrink(head,&candidate,hole->children+i,max_dim);
	    changed = 1;
	  }  
	}
	if(! changed) break;
      }
      break;
      
    case PARTIAL:
    case FULL21:
      //The construction of the intersection forbids this case
      Assert(0);
      break;
      

  }
  
  //This is a shitty corner case, that can occur.
  if(vBox(head,&candidate) <= abs(head->epsilon*vBox(head,&candidate))) goto nohole;
    
  //fprintf(stderr, "\nResized hole\n");
  //_printTree(head,&candidate,0);
  //See if we need to transfer children to the new hole
  //We run the loop backwards, because unregister child substitutes from the back
  for(i=hole->nr_children-1; i >=0 ; i--){
    intersection_t type = getIntersectionType(head,&candidate,hole->children+i);
    if(type == FULL12){
      registerChild(head,&candidate,hole->children+i);
      unregisterChild(head,hole,i);
    }
    else if(type == EQUALITY){
      Assert(0);
    }  
      
    Assert(type != PARTIAL);
  }
  
  
  candidate.tuples = hole->counter * (v(head,&candidate)/v_qib);
  candidate.counter = hole->counter * (v(head,&candidate)/v_qib);
  //fprintf(stderr,"%f * %f / %f",hole->counter,v(head,&candidate),v_qib);
  
  registerChild(head,hole,&candidate);
  
  Assert(_disjunctivenessTest(head,&candidate));
  Assert(_disjunctivenessTest(head,hole));
  
  parent_vol = hole->tuples - candidate.tuples;
  if(parent_vol >= 0){
    hole->tuples = parent_vol;
  }
  else {
    hole->tuples = 0;
  }  
  
  //Does this bucket still carry information?
  //If not, migrate all children to the parent. Of course, the root bucket can't be removed.
  if(parent != NULL && v(head,hole) <= abs(head->epsilon*vBox(head,hole))){
    int old_parent_size;
    tmp = *hole;
    unregisterChild(head,parent,pos);
    
    old_parent_size = parent->nr_children;
    for(i = 0; i < tmp.nr_children; i++){
      registerChild(head, parent, tmp.children+i);
    }
    
    for(i=old_parent_size; i < tmp.nr_children; i++){
      _drillHoles(head,parent,parent->children+i);
    }

    head->holes--;

    releaseRessources(&tmp);
    
    Assert(_disjunctivenessTest(head,parent));
    return;
  }
  head->holes++;
  
  //Tell the children about the new query
  old_hole_size = hole->nr_children;
  for(i=0; i < old_hole_size; i++){
    _drillHoles(head,hole,hole->children+i);
  } 
  return;
  
  
  nohole:
  releaseRessources(&candidate);
  old_hole_size = hole->nr_children;
  for(i=0; i < old_hole_size; i++){
    _drillHoles(head,hole,hole->children+i);
  }
  return;
}


static void drillHoles(st_head_t* head){
  _drillHoles(head, NULL, (st_hole_t*) head);
}

/**
 * Calculate the parent child merge penalty for a given pair of buckets
 */
static kde_float_t parentChildMergeCost(const st_head_t* head, const st_hole_t* parent, const st_hole_t* child){
  kde_float_t fbp, fbc, fbn, vbp, vbc, vbn;
  
  if(parent == NULL) return INFINITY;
    
  fbp = parent->tuples; 
  fbc = child->tuples;
  fbn = fbc + fbp;
  vbp = v(head,parent);
  vbc = v(head,child);
  vbn = vbp + vbc;
  
  return abs(fbp - fbn * vbp / vbn) + abs(fbc - fbn * vbc / vbn);
}

/**
 * Calculate the penalty for a parent double child merge(Corner case of a sibling sibling merge)
 */
static kde_float_t parentChildChildMergeCost(const st_head_t* head, const st_hole_t* parent, const st_hole_t* c1, const st_hole_t* c2){
  kde_float_t fbp, fbc1, fbc2, fbn, vbp, vbc1, vbc2, vbn;
  if(parent == NULL) return INFINITY;
    
  fbp = parent->tuples; 
  fbc1 = c1->tuples;
  fbc2 = c2->tuples;
  fbn = fbc1 + fbc2 + fbp;
  vbp = v(head,parent);
  vbc1 = v(head,c1);
  vbc2 = v(head,c2);
  vbn = vbp + vbc1 + vbc2;
  
  return abs(fbp - fbn * vbp / vbn) + abs(fbc1 - fbn * vbc1 / vbn) + abs(fbc2 - fbn * vbc2 / vbn);
}

/**
 * Apply a parent child merge to the histogram
 */
static void performParentChildMerge(st_head_t* head, st_hole_t* parent, st_hole_t* child){ 
  int i, pos = 0;
  parent->tuples += child->tuples;
  
  //Remeber the position because registerChild might invalidate the child
  pos = child-parent->children;
  
  //Migrate all children
  for(i = 0; i < child->nr_children; i++){
    registerChild(head,parent,child->children+i);
  }
  
  releaseRessources(parent->children+pos);
  unregisterChild(head,parent,pos);
 
  head->holes--;
}


/*
 * Apply a sibling sibling merge to the histogram
 */
static void performSiblingSiblingMerge(st_head_t* head, st_hole_t* parent, st_hole_t* c1,st_hole_t* c2){
    kde_float_t f_bn, f_bp, f_b1, f_b2, v_old, v_bp, v_b1, v_b2;
    
    st_hole_t bn = createNewSthole(head);
    
    int i = 0;
    //Keep a copy of c1 and c2. This make ist much easier to releaseRessources them later.
    st_hole_t c1_c = *c1;
    st_hole_t c2_c = *c2;
    intersection_t type;
    int pos1, pos2;
    
    f_b1 = c1 -> tuples;
    f_b2 = c2 -> tuples;
    f_bp = parent->tuples;
    v_bp = v(head, parent);
    v_b1 = v(head, c1);
    v_b2 = v(head, c2);
    
    

    
    boundingBox(head,c1,&bn);
    boundingBox(head,c2,&bn);
    
    //The mechanics of unregister child function
    //requires c1 <c2 
    if(c2 < c1){
	st_hole_t* tmp = c2;
	c2=c1;
	c1=tmp;
    }  
    
    //Remeber the positions beacause parent->children will be messed up
    pos2 = c2 - parent->children;
    pos1 = c1 - parent->children;
    
    unregisterChild(head,parent,pos2);
    unregisterChild(head,parent,pos1);
    
    //Progressively increase the size of the bounding box until we have no partial intersections
    for(;;){
      int changed = 0;
      for(i=0; i < parent->nr_children; i++){
	
	type = getIntersectionType(head,parent->children+i,&bn);
	if(type == PARTIAL){
	  //fprintf(stderr,"Adjusting partial \n");
	  boundingBox(head,parent->children+i,&bn);
	  changed = 1;
	}
      }
      if(! changed) break;
    }
    
    //A final pass to add all enclosed boxes
    for(i=parent->nr_children-1; i >=0 ; i--){
      type = getIntersectionType(head,parent->children+i,&bn);
      if(type == FULL21){
	registerChild(head,&bn,parent->children+i);
	unregisterChild(head,parent,i);
      }
    }
    
    //p-c2 merge corner case
    if(getIntersectionType(head,parent,&bn) == EQUALITY){
      //performParentChild will change the underlying memory of the children.

      performParentChildMerge(head,parent,parent->children+pos2);
      performParentChildMerge(head,parent,parent->children+pos1);
      releaseRessources(&bn);
      return;
    }
    
    //v(head,&bn) = vBox(head) - vBox(participants)
    v_old = v(head,&bn) - vBox(head,&c1_c) - vBox(head,&c2_c);
    if(v_old < 0 ){
      v_old = 0;
    }
    
    f_bn = f_b1 + f_b2 + f_bp * v_old / v_bp;
    if(f_bn < 0 ){
      f_bn = 0;
    }
    
    bn.tuples = f_bn;
    parent->tuples = parent->tuples * (1 - v_old/v_bp);
    
    
    for(i = 0; i < c2_c.nr_children; i++){
      registerChild(head,&bn,c2_c.children+i);
    }
    
    for(i = 0; i < c1_c.nr_children; i++){
      registerChild(head,&bn,c1_c.children+i);
    }
    
    releaseRessources(&c1_c);
    releaseRessources(&c2_c);
    
    registerChild(head,parent,&bn);
    head->holes--;;
}  

/**
 * Calculate the cost for an sibling sibling merge
 */
static kde_float_t siblingSiblingMergeCost(const st_head_t* head, const st_hole_t* parent, const st_hole_t* c1, const st_hole_t* c2){    
    st_hole_t bn = createNewSthole(head);
    int i = 0;
    intersection_t type;
    kde_float_t f_bn, f_bp, f_b1, f_b2, v_old, v_bp, v_bn, v_b1, v_b2;
    
    f_b1 = c1 -> tuples;
    f_b2 = c2 -> tuples;
    f_bp = parent->tuples;
    v_bp = v(head, parent);
    v_b1 = v(head, c1);
    v_b2 = v(head, c2);
    
    boundingBox(head,c1,&bn);
    boundingBox(head,c2,&bn);
    
    //Progressively increase the size of the bounding box until we have no partial intersections
    for(;;){
      int changed = 0;
      for(i=0; i < parent->nr_children; i++){
	type = getIntersectionType(head,parent->children+i,&bn);
	if(type == PARTIAL){  
	  boundingBox(head,parent->children+i,&bn);
	  changed = 1;
	}
      }
      if(! changed) break;
    }
    
    //A final pass to add all enclosed boxes
    for(i=0; i < parent->nr_children; i++){
      intersection_t type = getIntersectionType(head,parent->children+i,&bn);
      if(parent->children+i == c1 || parent->children+i == c2) continue;
      if(type == FULL21){
	registerChild(head,&bn,parent->children+i);
      }
    }
    
    if(getIntersectionType(head,parent,&bn) == EQUALITY){
      releaseRessources(&bn);
      return parentChildChildMergeCost(head,parent,c1,c2);
    }
    
    //v(head,&bn) = vBox(bn) - vBox(participants)
    v_old = v(head,&bn) - vBox(head,c1) - vBox(head,c2);

    releaseRessources(&bn);
    
    f_bn = f_b1 + f_b2 + f_bp * v_old / v_bp;
    
    //Caution: there is an error in the full paper saying this is the value for v_p.
    //It obviously is v_bn.
    v_bn = v_old + v_b1 + v_b2;
    
    return abs(f_bn * v_old / v_bn - f_bp * v_old / v_bp)
	   + abs(f_b1 - f_bn * v_b1 / v_bn )
	   + abs(f_b2 - f_bn * v_b2 / v_bn );    
}  


typedef struct merge {
  st_hole_t* parent;
  st_hole_t* child1;
  st_hole_t* child2;
  kde_float_t penalty;
} merge_t;  

/**
 * Find a min cost merge in the tree
 */ 
static void _getSmallestMerge(st_head_t* head, st_hole_t* parent, st_hole_t* hole, merge_t* bestMerge){
  int i = 0;
  kde_float_t test = 0.0;
  
  //Ask all children for the best value in their branch
  for(i = 0; i < hole->nr_children; i++){
    _getSmallestMerge(head,hole,hole->children+i,bestMerge);
  }  
  
  //The head node can't be the child in a merge and has no siblings.
  if(parent == NULL) return;
  
  //Test parent child merge cost
  test = parentChildMergeCost(head,parent,hole);
  if(test < bestMerge->penalty){
      bestMerge->penalty = test;
      bestMerge->parent = parent;
      bestMerge->child1 = hole;
      bestMerge->child2 = NULL;
  }
  
  //Test sibling sibling merge cost to all siblings
  for(i = 0; i < parent->nr_children; i++){
    if(parent->children + i == hole) continue;
    test = siblingSiblingMergeCost(head,parent,hole,parent->children + i);
    if(test < bestMerge->penalty){
      bestMerge->child1 = hole;
      bestMerge->child2 = parent->children +i;
      bestMerge->parent = parent;
      bestMerge->penalty = test;
    }
  }
}

/**
 * Dumps the complete histogram to stdout
 */ 
static void printTree(st_head_t* head){
  fprintf(stderr,"Dimensions: %u\n",head->dimensions);
  fprintf(stderr,"Max #holes: %i\n",head->max_holes);
  fprintf(stderr,"Current #holes: %i\n",head->holes);
  _printTree(head, (st_hole_t*) head,0);
}

/**
 * Performs min penalty merges until we are in our memory boundaries again.
 */ 
static void mergeHoles(st_head_t* head){
    while(head->holes > head->max_holes){
      //fprintf(stderr,"Holes: %i > %i\n",head->holes, head->max_holes);
      merge_t bestMerge;
      bestMerge.penalty = INFINITY;
      
      //Ask for the best possible merge in the tree
      _getSmallestMerge(head, NULL, (st_hole_t*) head, &bestMerge);
      
      //See, what kind of merge it is and perform it
      if(bestMerge.child2 != NULL){
	//fprintf(stderr,"Sibling-sibling merge!\n");
 	performSiblingSiblingMerge(head, bestMerge.parent, bestMerge.child1, bestMerge.child2);
      }
      else {
	//fprintf(stderr,"Parent-child merge!\n");
	performParentChildMerge(head,bestMerge.parent,bestMerge.child1);
      }
    }  
}  

/**
 * Is called for every qualifying tuple processed in a sequential scan
 * and then finds the correct bucket and increases the counter 
*/
static void propagateTuple(st_head_t* head, Relation rel, const TupleTableSlot* slot){
  
  int i = 0;
  int rc = 0;
  bool isNull;
  TupleDesc desc = slot->tts_tupleDescriptor;
  kde_float_t* tuple = (kde_float_t*) malloc(sizeof(kde_float_t)*head->dimensions);
  
  HeapTuple htup = slot->tts_tuple;
  Assert(htup);

  for(i = 0; i < desc->natts; i++){
    Datum datum;
    
    //This should skip columns that are requested but not handled by our estimator.
    if(! (head->columns & (0x1 << desc->attrs[i]->attnum))) continue;
    
    datum = heap_getattr(htup, desc->attrs[i]->attnum ,RelationGetDescr(rel), &isNull);
    
    Assert(desc->attrs[i]->atttypid == FLOAT8OID || 
      desc->attrs[i]->atttypid == FLOAT4OID
    );
    
    
    if(desc->attrs[i]->atttypid == FLOAT8OID){
      tuple[head->column_order[desc->attrs[i]->attnum]] = (kde_float_t) DatumGetFloat8(datum);
    }
    else {
      tuple[head->column_order[desc->attrs[i]->attnum]] = (kde_float_t) DatumGetFloat4(datum);
    }
    
  } 
  
  //Very well, by now we should have a nice tuple. Lets do some traversing.
  rc = _propagateTuple(head,(st_hole_t*) head,tuple);
  Assert(rc); //We should never encounter a tuple that does not fit our relation
}

/**
 * API method to propagate a tuple from the result stream.
 */
void stholes_propagateTuple(Relation rel, const TupleTableSlot* slot){
  if(current != NULL && rel->rd_id == current->table){
    return propagateTuple(current,rel,slot);
  }
}  

/**
 * API method to create a new histogram and remove the old one.
 */
void stholes_addhistogram(Oid table,AttrNumber* attributes,unsigned int dimensions){
  if(current != NULL) destroyHistogram(current);
  current = createNewHistogram(table,attributes,dimensions);
  fprintf(stderr, "Created a new stholes histogram\n");
  
}  

static int est(st_head_t* head, const ocl_estimator_request_t* request, Selectivity* selectivity){
  int request_columns = 0;
  int i = 0;
  kde_float_t ivol;
  
  //Can we answer this query?
  for (i = 0; i < request->range_count; ++i) {
    request_columns |= 0x1 << request->ranges[i].colno;
  }

  //We do not allow queries missing restrictions on a variable
  if ((head->columns | request_columns) != head->columns || request->range_count != head->dimensions) return 0;

  //Bring the request in a nicer form and store it
  setLastQuery(head,request);
  
  //"If the current query q extends the beyond the boundaries of the root bucket
  //we expand the root bucket so that it covers q." Section 4.2, p. 8
  //We usually assume closed intervalls for the queries, but stholes bounds are half open intervalls (5.2).
  //We add a machine epsilon to the bound, just in case
  for(i = 0; i < head->dimensions; i++){  
    if(((st_hole_t*) head)->bounds[2*i] > head->last_query.bounds[2*i])
      ((st_hole_t*) head)->bounds[2*i] = head->last_query.bounds[2*i];
    if(((st_hole_t*) head)->bounds[2*i+1] < head->last_query.bounds[2*i+1])
      ((st_hole_t*) head)->bounds[2*i+1] = head->last_query.bounds[2*i+1]; //+ abs(head->last_query.bounds[2*i+1]) * head->epsilon;  
  }

  
  *selectivity = _est(head, (const st_hole_t*) head,&ivol);
  return 1;
} 


static void _printTree(st_head_t* head, st_hole_t* hole, int depth){
  int i = 0;
  for(i=0; i < depth; i++){
    fprintf(stderr,"\t");
  }
  
  for(i = 0; i < head->dimensions; i++){
    fprintf(stderr,"[%f , %f] ", hole->bounds[2*i], hole->bounds[2*i+1]);
  }
  fprintf(stderr,"Counter %f",hole->counter);
  fprintf(stderr," Tuples %f",hole->tuples);
  fprintf(stderr,"\n");
  for(i = 0; i < hole->nr_children; i++){
    _printTree(head,hole->children+i,depth+1);
  }  
}  

/**
 * API method called by the optimizer to retrieve a selectivity estimation.
 */
int stholes_est(Oid rel, const ocl_estimator_request_t* request, Selectivity* selectivity){
  int rc;
  if(current == NULL) return 0;
  
  if(rel == current->table){
    resetAllCounters((st_hole_t*) current);
    rc = est(current, request,selectivity);
    current->last_selectivity = *selectivity;
    current->process_feedback = 1;
    return rc;
  }
  
  return 0;
}  
 
  
/**
 *  API method called to initiate the histogram optimization process
 */
void stholes_process_feedback(PlanState *node){
  if(current == NULL) return;
  if(nodeTag(node) != T_SeqScanState) return;
  if(node->instrument == NULL || node->instrument->kde_rq == NULL) return;
  
  //fprintf(stderr, "Process feedback! %i %i\n",((SeqScanState*) node)->ss_currentRelation->rd_id == current->table,current->process_feedback);
  
  if(((SeqScanState*) node)->ss_currentRelation->rd_id == current->table && current->process_feedback){
    float8 qual_tuples = (float8)(node->instrument->tuplecount + node->instrument->ntuples)/(node->instrument->nloops+1);
    float8 all_tuples = (float8)(node->instrument->tuplecount + node->instrument->nfiltered2 + node->instrument->nfiltered1 + node->instrument->ntuples)/(node->instrument->nloops+1);
   
    
    drillHoles(current);
    mergeHoles(current);
    
    ocl_reportErrorToLogFile(((SeqScanState*) node)->ss_currentRelation->rd_id,qual_tuples / all_tuples,current->last_selectivity/all_tuples, all_tuples);
    
    current->process_feedback = 0;
  }
  
}    




