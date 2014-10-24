#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"

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

/**
 * An sthole instance. 
 * Contains the number of tuples it contains, its bounds and its children
 */ 
typedef struct st_hole {
  kde_float_t tuples;
  // Children of this hole.
  int nr_children;
  int child_capacity;
  struct st_hole* children;
  // Boundaries of this hole.
  kde_float_t* bounds;
  // Working counter for the statistics step.
  kde_float_t counter;
  // Cache for volume computations.
  kde_float_t v;
  kde_float_t v_box;
  // Cache for merge costs.
  kde_float_t* children_merge_cost;
  char children_merge_cost_cache_dirty;
} st_hole_t;


/**
 * The head of an stholes histogram. 
 * Contains additional meta information
 */ 
typedef struct st_head {
  // Root hole.
  st_hole_t root;
  // Meta information.
  int holes;
  int max_holes;
  unsigned int dimensions;
  Oid table;
  int32 columns;
  unsigned int* column_order;
  // Information for updating the structure.
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
static void initializeNewSthole(st_hole_t* hole, const st_head_t* head) {
  // Zero-initialize everything.
  memset(hole, 0, sizeof(st_hole_t));
  // Now allocate the bounds and initialize them with +/- infinity.
  hole->bounds = (kde_float_t*) malloc(sizeof(kde_float_t)*head->dimensions*2);
  int i = 0;
  for (; i<head->dimensions ; i++){
    hole->bounds[i*2] = INFINITY;
    hole->bounds[i*2+1] = -INFINITY;
  }
  // Initialize the cached computations.
  hole->v = -1.0f;
  hole->v_box = -1.0f;
}

/**
 * Create a new head bucket for the histogram
 */
static st_head_t* createNewHistogram(
    Oid table, AttrNumber* attributes, unsigned int dimensions){
  st_head_t* head = (st_head_t*) calloc(1,sizeof(st_head_t));
  
  head->dimensions = dimensions;
  head->table = table;
  head->process_feedback = 0;
  head->max_holes = stholes_hole_limit;
  head->holes = 1;
  head->column_order = calloc(1, 32 * sizeof(unsigned int));
  
  initializeNewSthole(&(head->last_query), head);
  initializeNewSthole(&(head->root), head);
  
  if(sizeof(kde_float_t) == sizeof(double)){
    head->epsilon = DBL_EPSILON;
  } else {
    head->epsilon = FLT_EPSILON;
  }
  
  int i = 0;
  for (; i<dimensions; ++i) {
     head->columns |= 0x1 << attributes[i];
     head->column_order[attributes[i]] = i;
  }
  
  //Initialize bound with +/- infinfity
  for(i=0 ; i<dimensions ; i++) {
    head->root.bounds[i*2] = INFINITY;
    head->root.bounds[i*2+1] = -INFINITY;
  }
  
  return head;
}

/**
 * Release the resources of an sthole
 */
static void releaseResources(st_hole_t* hole) {
  free(hole->bounds);
  free(hole->children);
  free(hole->children_merge_cost);
}

static void _destroyHistogram(st_hole_t* hole){
  int i = 0;
  for (; i < hole->nr_children; i++) {
    _destroyHistogram(hole->children + i);
  }
  releaseResources(hole);
}

static void destroyHistogram(st_head_t* head){
  _destroyHistogram(&(head->root));
  free(head);
}


// Convert the last query to an sthole.
// We can then simply use our standard functions for vBox and v for it.
static void setLastQuery(
    st_head_t* head, const ocl_estimator_request_t* request) {
  int i = 0;
  for(; i < request->range_count; i++){
    // Add tiny little epsilons, if necessary, to account for the [) buckets.
    if (request->ranges[i].lower_included) {
      head->last_query.bounds[head->column_order[request->ranges[i].colno]*2] =
          request->ranges[i].lower_bound;
    } else {
      head->last_query.bounds[head->column_order[request->ranges[i].colno]*2] =
          request->ranges[i].lower_bound +
          abs(request->ranges[i].lower_bound) * head->epsilon;
    }
    if (request->ranges[i].upper_included) {
      head->last_query.bounds[head->column_order[request->ranges[i].colno]*2+1] =
          request->ranges[i].upper_bound +
          abs(request->ranges[i].upper_bound) * head->epsilon;
    } else {
      head->last_query.bounds[head->column_order[request->ranges[i].colno]*2+1] =
          request->ranges[i].upper_bound;
    }
  }
  // Invalidate the volume cache.
  head->last_query.v = -1.0f;
  head->last_query.v_box = -1.0f;
}


/**
 * Add a new child to the given parent.
 */
static void registerChild(
    const st_head_t* head, st_hole_t* parent, const st_hole_t* child) {
  parent->nr_children++;
  if (parent->child_capacity < parent->nr_children) {
    // Increase the child capacity by 10.
    parent->child_capacity += 10;
    parent->children = realloc(
        parent->children, parent->child_capacity * sizeof(st_hole_t));
  }
  parent->children[parent->nr_children-1] = *child;
  // We changed the children, invalidate the volume cache.
  parent->v = -1.0f;
  // Invalidate the children merge cost cache.
  free(parent->children_merge_cost);
  parent->children_merge_cost = NULL;
}

/**
 * Remove the child at position pos in bucket parent
 */
static void unregisterChild(st_head_t* head, st_hole_t* parent, int pos) {
  Assert(pos >= 0);
  Assert(pos < parent->nr_children);
  if (pos != parent->nr_children-1) {
    parent->children[pos] = parent->children[parent->nr_children-1];
  }
  parent->nr_children--;
  // We changed the children, invalidate the volume cache.
  parent->v = -1.0f;
  // Invalidate the children merge cost cache.
  free(parent->children_merge_cost);
  parent->children_merge_cost = NULL;
}

/**
 * vBox operator from the paper
 * bucket volume (children included)
 */
static kde_float_t vBox(const st_head_t* head, st_hole_t* hole) {
  if (hole->v_box < 0) {
    // No valid cached value, recompute the volume.
    kde_float_t volume = 1.0;
    int i = 0;
    for (; i < head->dimensions; i++) {
      volume *= hole->bounds[i*2+1] - hole->bounds[i*2];
    }
    hole->v_box = fmax(volume, 0);
  }
  // Return the cached volume.
  return hole->v_box;
}

/**
 * v function from the paper
 * bucket volume (children excluded)
 */
static kde_float_t v(const st_head_t* head, st_hole_t* hole) {
  if (hole->v < 0) {
    // No valid cached value, recompute the volume.
    kde_float_t v = vBox(head, hole);
    int i = 0;
    for (; i < hole->nr_children; i++){
      v -= vBox(head, hole->children + i);
    }
    Assert(v >= -1.0); // This usually means that something went terribly wrong.
    hole->v = fmax(0.0, v);
  }
  return hole->v;
}

/**
 * Calculate the intersection bucket with the last query (head->lastquery).
 * Stores it in target_hole
 */
static void intersectWithLastQuery(
    const st_head_t* head, const st_hole_t* hole, st_hole_t* target_hole) {
  int i = 0;
  for (; i < head->dimensions; i++) {
    target_hole->bounds[2*i] = fmax(
        head->last_query.bounds[2*i], hole->bounds[2*i]);
    target_hole->bounds[2*i+1] = fmin(
        head->last_query.bounds[2*i+1], hole->bounds[2*i+1]);
  }
  // The volume of the target hole has changed, invalid the cached volume.
  target_hole->v = -1.0f;
  target_hole->v_box = -1.0f;
}

/**
 * Calculate the smallest box containing hole and target_hole.
 * Stores the result in target_hole
 */
static void boundingBox(
    const st_head_t* head, const st_hole_t* hole, st_hole_t* target_hole) {
  int i = 0;
  for (; i < head->dimensions; i++) {
    target_hole->bounds[2*i] = fmin(
        target_hole->bounds[2*i], hole->bounds[2*i]);
    target_hole->bounds[2*i+1] = fmax(
        target_hole->bounds[2*i+1], hole->bounds[2*i+1]);
  }
  // The volume of the target hole has changed, invalid the cached volume.
  target_hole->v = -1.0f;
  target_hole->v_box = -1.0f;
}

typedef enum {FULL12, FULL21, NONE, PARTIAL, EQUALITY} intersection_t;

/* Calculates the strongest relationship between two histogram buckets
 * EQUALITY: 	hole1 and two are the same
 * FULL12: 	hole1 fully contains hole2
 * FULL21: 	hole2 fully contains hole1
 * PARTIAL:	The holes have a partial intersection
 * NONE:	The holes are disjunct
 */

static intersection_t getIntersectionType(
    const st_head_t* head, const st_hole_t* hole1, const st_hole_t* hole2) {
  int enclosed12 = 1;
  int enclosed21 = 1;

  int i = 0;
  for(; i < head->dimensions; i++){
    //Case 1: We have no intersection with this hole
    //If this does not intersect with one of the intervals of the box, we have nothing to do.
    //Neither have our children.
    if (hole1->bounds[2*i+1] <= hole2->bounds[2*i] || hole1->bounds[2*i] >= hole2->bounds[2*i+1]) {
      return NONE;
    }

    if (!(hole2->bounds[2*i] >= hole1->bounds[2*i] && hole2->bounds[2*i+1] <= hole1->bounds[2*i+1])) {
      enclosed12 = 0;
    }

    if(!(hole1->bounds[2*i] >= hole2->bounds[2*i] && hole1->bounds[2*i+1] <= hole2->bounds[2*i+1])) {
      enclosed21 = 0;
    }
  }
  if (enclosed21 && enclosed12){
    return EQUALITY;
  } else if(enclosed21){
    return FULL21;
  } else if(enclosed12){
    return FULL12;
  } else {
    return PARTIAL;
  }
}

/** 
 * Debugging function, can be used to check the histogram for inconsistencies
 * regarding the disjunctiveness of buckets.
 */
static int _disjunctivenessTest(st_head_t* head, st_hole_t* hole) {
  int i,j = 0;
  for (; i < hole->nr_children; i++) {
    for (j = i+1; j < hole->nr_children; j++) {
      if (getIntersectionType(
           head, hole->children+i, hole->children+j) != NONE) {
        fprintf(stderr, "Intersection between child %i and %i is %i\n", i, j,
                getIntersectionType(head,hole->children+i,hole->children+j));
        return 0;
      }
    }
  }
  return 1;
}

/**
 * Aggregate estimated tuples recursively
 */
static kde_float_t _est(
    const st_head_t* head, const st_hole_t* hole,
    kde_float_t* intersection_vol) {
  kde_float_t est = 0.0;
  *intersection_vol = 0;
  
  // If we don't intersect with the query, the estimate is zero.
  if (getIntersectionType(head, hole, &head->last_query) == NONE) {
    return est;
  }
  
  //  Calculate v(q b)
  st_hole_t q_i_b;
  initializeNewSthole(&q_i_b, head);
  intersectWithLastQuery(head, hole, &q_i_b);
  *intersection_vol = vBox(head, &q_i_b);
  
  kde_float_t v_q_i_b = *intersection_vol;
  
  // Recurse over all children to sum up estimates and remove child volumes.
  int i = 0;
  for (; i < hole->nr_children; i++) {
    kde_float_t child_intersection;
    est += _est(head, hole->children + i, &child_intersection);
    v_q_i_b -= child_intersection;
  }
  // Ensure that v_q_i_b is capped by zero.
  v_q_i_b = fmax(0.0, v_q_i_b);
  
  // If the hole is overly filled, we might run into numerical issues here
  // that lead to empty estimates. Therefore we only add up the estimates
  // for holes that we consider safe.
  kde_float_t vh = v(head, hole);
  if (vh >= abs(head->epsilon * vBox(head, hole))) {
    est += hole->tuples * (v_q_i_b / vh); 
  }
  
  releaseResources(&q_i_b);
  
  return est;
}  


static int _propagateTuple(
    st_head_t* head, st_hole_t* hole, kde_float_t* tuple) {
  // Check if this point is within our bounds.
  int i = 0;
  for (; i < head->dimensions; i++) {
      // If it is not, our father will hear about this
      if (tuple[i] >= hole->bounds[2*i+1] ||
          tuple[i] < hole->bounds[2*i]) {
        return 0;
      }
  }
  // Inform our children about this point
  for (i = 0; i < hole->nr_children; i++) {
    // If one of our children claims this point, we tell our parents about it.
    if (_propagateTuple(head, hole->children + i, tuple) == 1) return 1;
  }
  
  // None of the children showed interest in the point, claim it for ourselves.
  hole->counter++;
  return 1;
}

/**
 * Recursively reset the counter for all holes 
 */
static void resetAllCounters(st_hole_t* hole) {
  hole->counter = 0;
  int i = 0;
  for (; i < hole->nr_children; i++) {
    resetAllCounters(hole->children + i);
  }
}

// Get the volume of intersection when shrinking it along dimension such that
// it does not intersect with the hole.
static kde_float_t getReducedVolume(
    st_head_t* head, st_hole_t* intersection, const st_hole_t* hole,
    int dimension) {
  
  // Compute the volume without the selected dimension.
  kde_float_t vol = vBox(head, intersection);
  vol /= intersection->bounds[dimension*2+1] -
         intersection->bounds[dimension*2];

  if (intersection->bounds[dimension*2] >= hole->bounds[dimension*2] &&
      intersection->bounds[dimension*2] < hole->bounds[dimension*2+1]) {
    // If the dimension is completely located inside the box, we cannot reduce
    // along this dimension.
    if (intersection->bounds[dimension*2+1] <= hole->bounds[dimension*2+1]) {
      return -INFINITY;
    }
    //If the lower bound is located inside the other box, we have to exchange it.
    return vol *
        (intersection->bounds[dimension*2+1] - hole->bounds[dimension*2+1]);
  } else if (
      intersection->bounds[dimension*2+1] > hole->bounds[dimension*2] &&
      intersection->bounds[dimension*2+1] <= hole->bounds[dimension*2+1]) {
    return vol *
        (hole->bounds[dimension*2] - intersection->bounds[dimension*2]);
  } else if (
      hole->bounds[dimension*2] >= intersection->bounds[dimension*2] &&
      hole->bounds[dimension*2+1] <= intersection->bounds[dimension*2+1]) {
    // The hole is completely in the intersection. In this case, we can either
    // reduce the lower or the upper bound.
    return fmax(
        vol*(hole->bounds[dimension*2]-intersection->bounds[dimension*2]),
        vol*(intersection->bounds[dimension*2+1]-hole->bounds[dimension*2+1]));
  } else {
    // If the dimensions are completely distinct, we should not have
    // called this method.
    //_printTree(head,intersection,0);
    //_printTree(head,hole,0);
    Assert(0);
    return 0.0;
  }
}

static void shrink(
    st_head_t* head, st_hole_t* intersection, const st_hole_t* hole,
    int dimension) {
  if (intersection->bounds[dimension*2] >= hole->bounds[dimension*2] &&
      intersection->bounds[dimension*2] < hole->bounds[dimension*2+1]) {
    // If the dimension is completely located inside the box, the dimension
    // is not eligible for dimension reduction.
    if (intersection->bounds[dimension*2+1] <= hole->bounds[dimension*2+1]) {
      Assert(0); //Not here.
    }
    // If the lower bound is located inside the other box, we have to change it.
    intersection->bounds[dimension*2] = hole->bounds[dimension*2+1];
  } else if (
      intersection->bounds[dimension*2+1] > hole->bounds[dimension*2] &&
      intersection->bounds[dimension*2+1] <= hole->bounds[dimension*2+1]) {
    intersection->bounds[dimension*2+1] = hole->bounds[dimension*2];
  } else if (
      hole->bounds[dimension*2] >= intersection->bounds[dimension*2] &&
      hole->bounds[dimension*2+1] <= intersection->bounds[dimension*2+1]) {
      if ((hole->bounds[dimension*2] - intersection->bounds[dimension*2]) >
          (intersection->bounds[dimension*2+1] - hole->bounds[dimension*2+1])) {
      intersection->bounds[dimension*2+1] = hole->bounds[dimension*2];
    } else {
      intersection->bounds[dimension*2] = hole->bounds[dimension*2+1];
    }
  } else {
    //If not at least one of the coordinates is located inside the box,
    // we should not have called this method.
    Assert(0);
  }
  // The volume has changed. Remove all cached values.
  intersection->v = -1.0f;
  intersection->v_box = -1.0f;
}  

// Returns the dimension that offers the least reduced volume when
// progressively shrinking intersection such that it does not partially
// intersect with the hole anymore.
static unsigned int minReducedVolumeDimension(
    st_head_t* head, st_hole_t* intersection, const st_hole_t* hole){
  int max_dim = -1;
  kde_float_t max_vol = -INFINITY;
  int i = 0;
  for(; i < head->dimensions; i++) {
    kde_float_t vol = getReducedVolume(head, intersection, hole, i);
    if(vol > max_vol) {
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
static void _drillHoles(st_head_t* head, st_hole_t* parent, st_hole_t* hole) {
  st_hole_t candidate, tmp;

  int i, pos, old_hole_size = 0;
  intersection_t type;
  kde_float_t v_qib;
  kde_float_t parent_vol;
  
  if (getIntersectionType(head, &(head->last_query), hole) == NONE) return;
  if (parent != NULL) pos = hole - parent->children; // Remember the position of the hole before we mess with the array
  initializeNewSthole(&candidate, head);
  initializeNewSthole(&tmp, head);
  // Get the intersection with the last query for this hole.
  intersectWithLastQuery(head, hole, &candidate);
  v_qib = vBox(head, &candidate); //*Will be adjusted to the correct value later
  type = getIntersectionType(head, hole, &candidate);
  
  // Shrink the candidate hole.
  switch(type){
    case NONE:
      return;
    
    //Case 2: We have complete intersection with this hole. Update stats.
    case EQUALITY:
      hole->tuples = hole->counter;
      goto nohole;     
      
    
    case FULL12:
      //We will use the tmp
      for (i = 0; i < hole->nr_children; i++) {
        if (getIntersectionType(head,&(head->last_query),hole->children+i) != NONE) {
          intersectWithLastQuery(head, hole->children+i, &tmp);
          v_qib -= vBox(head,&tmp);
        }
      } 
      releaseResources(&tmp);
      // Shrink the candidate until it does not intersect any children.
      while (true) {
        int changed = 0;
        for (i = 0; i < hole->nr_children; i++){
          // Full intersections:
          intersection_t type = getIntersectionType(
              head, hole->children+i, &candidate);
          // Full and no intersection are no problem:
          if (type == FULL12) {
            goto nohole;
          } else if(type == NONE) {
            continue;
          } else if(type == EQUALITY){
            //The child can handle this case on its own.
            goto nohole;
          } else if(type == FULL21){
            continue; //The child is completely located inside the intersection.
          } else {
            unsigned int max_dim =
                minReducedVolumeDimension(head, &candidate, hole->children+i);
            shrink(head, &candidate, hole->children+i, max_dim);
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
  if (vBox(head, &candidate) <= abs(head->epsilon*vBox(head,&candidate))) {
    goto nohole;
  }
    
  // See if we need to transfer children to the new hole
  // We run the loop backwards, because unregister child substitutes
  // from the back.
  for (i=hole->nr_children-1; i >=0 ; i--){
    intersection_t type = getIntersectionType(
        head, &candidate, hole->children+i);
    if (type == FULL12) {
      registerChild(head, &candidate, hole->children + i);
      unregisterChild(head, hole, i);
    } else if(type == EQUALITY) {
      Assert(0);
    }
    Assert(type != PARTIAL);
  }
  
  candidate.tuples = hole->counter * (v(head,&candidate)/v_qib);
  candidate.counter = hole->counter * (v(head,&candidate)/v_qib);
  // Finally, register the candidate hole.
  registerChild(head, hole, &candidate);
  
  Assert(_disjunctivenessTest(head, &candidate));
  Assert(_disjunctivenessTest(head, hole));
  
  parent_vol = hole->tuples - candidate.tuples;
  if (parent_vol >= 0) {
    hole->tuples = parent_vol;
  } else {
    hole->tuples = 0;
  }  
  
  // Does this bucket still carry information?
  // If not, migrate all children to the parent. Of course, the root bucket can't be removed.
  if (parent != NULL && v(head,hole) <= abs(head->epsilon*vBox(head,hole))) {
    int old_parent_size;
    tmp = *hole;
    unregisterChild(head, parent, pos);
    
    old_parent_size = parent->nr_children;
    for (i = 0; i < tmp.nr_children; i++){
      registerChild(head, parent, tmp.children+i);
    }
    
    for (i=old_parent_size; i < tmp.nr_children; i++){
      _drillHoles(head, parent, parent->children+i);
    }

    head->holes--;

    releaseResources(&tmp);
    
    Assert(_disjunctivenessTest(head,parent));
    return;
  }
  head->holes++;
  
  // Tell the children about the new query.
  old_hole_size = hole->nr_children;
  for (i=0; i < old_hole_size; i++) {
    _drillHoles(head, hole, hole->children+i);
  } 
  return;
  
nohole:
  releaseResources(&candidate);
  old_hole_size = hole->nr_children;
  for (i=0; i < old_hole_size; i++) {
    _drillHoles(head, hole, hole->children+i);
  }
  return;
}


static void drillHoles(st_head_t* head) {
  _drillHoles(head, NULL, &(head->root));
}

/**
 * Calculate the parent child merge penalty for a given pair of buckets
 */
static kde_float_t parentChildMergeCost(
    const st_head_t* head, const st_hole_t* parent, const st_hole_t* child) {
  if(parent == NULL) return INFINITY;
    
  kde_float_t fbp = parent->tuples;
  kde_float_t fbc = child->tuples;
  kde_float_t fbn = fbc + fbp;
  kde_float_t vbp = v(head, parent);
  kde_float_t vbc = v(head, child);
  kde_float_t vbn = vbp + vbc;
  
  return abs(fbp - fbn * vbp / vbn) + abs(fbc - fbn * vbc / vbn);
}

/**
 * Calculate the penalty for a parent double child merge (Corner case
 * of a sibling sibling merge)
 */
static kde_float_t parentChildChildMergeCost(
    const st_head_t* head, const st_hole_t* parent,
    const st_hole_t* c1, const st_hole_t* c2) {
  if(parent == NULL) return INFINITY;
    
  kde_float_t fbp = parent->tuples;
  kde_float_t fbc1 = c1->tuples;
  kde_float_t fbc2 = c2->tuples;
  kde_float_t fbn = fbc1 + fbc2 + fbp;
  kde_float_t vbp = v(head, parent);
  kde_float_t vbc1 = v(head, c1);
  kde_float_t vbc2 = v(head, c2);
  kde_float_t vbn = vbp + vbc1 + vbc2;
  
  return abs(fbp - fbn * vbp / vbn) +
         abs(fbc1 - fbn * vbc1 / vbn) +
         abs(fbc2 - fbn * vbc2 / vbn);
}

/**
 * Apply a parent child merge to the histogram
 */
static void performParentChildMerge(
    st_head_t* head, st_hole_t* parent, st_hole_t* child){
  int pos = 0;
  parent->tuples += child->tuples;
  
  // Remember some info about the child, since registerChild might realloc the
  // parent's children buffer, which leads to our child pointer becoming
  // invalid.
  int pos_in_parent_child_array = child - parent->children;
  st_hole_t* child_children = child->children;
  unsigned int child_nr_children = child->nr_children;
  

  // Migrate all children.
  int i = 0;
  for (; i < child_nr_children; i++) {
    registerChild(head, parent, child_children + i);
  }
  
  releaseResources(parent->children + pos);
  unregisterChild(head, parent, pos);
 
  head->holes--;
}

/**
 * Calculate the cost for an sibling sibling merge
 */
static kde_float_t siblingSiblingMergeCost(
    const st_head_t* head, const st_hole_t* parent,
    const st_hole_t* c1, const st_hole_t* c2) {
  int i;
  st_hole_t bn;
  initializeNewSthole(&bn, head);

  kde_float_t f_b1 = c1->tuples;
  kde_float_t f_b2 = c2->tuples;
  kde_float_t f_bp = parent->tuples;
  kde_float_t v_bp = v(head, parent);
  kde_float_t v_b1 = v(head, c1);
  kde_float_t v_b2 = v(head, c2);

  boundingBox(head, c1, &bn);
  boundingBox(head, c2, &bn);

  // Progressively increase the size of the bounding box until we have
  // no partial intersections.
  while (true) {
    int changed = 0;
    for (i = 0; i < parent->nr_children; i++) {
      st_hole_t* child = parent->children + i;
      if (child == c1 || child == c2) continue; // Skip to-be-merged children.
      // Now check whether bn intersects with this child. If it does, we have
      // to grow bn until this intersection disappears.
      intersection_t type = getIntersectionType(head, child, &bn);
      if (type == PARTIAL) {
        boundingBox(head, child, &bn);
        changed = 1;
      }
    }
    if (!changed) break;
  }

  // A final pass to add all enclosed boxes.
  for (i=0; i < parent->nr_children; i++) {
    intersection_t type = getIntersectionType(head,parent->children+i,&bn);
    if (parent->children+i == c1 || parent->children+i == c2) continue;
    if (type == FULL21) {
      registerChild(head, &bn, parent->children+i);
    }
  }

  if (getIntersectionType(head,parent,&bn) == EQUALITY) {
    releaseResources(&bn);
    return parentChildChildMergeCost(head,parent,c1,c2);
  }

  //v(head,&bn) = vBox(bn) - vBox(participants)
  kde_float_t v_old = v(head,&bn) - vBox(head,c1) - vBox(head,c2);

  releaseResources(&bn);
  kde_float_t f_bn = f_b1 + f_b2 + f_bp * v_old / v_bp;

  // Caution: there is an error in the full paper saying this is the value for
  // v_p. It obviously is v_bn.
  kde_float_t v_bn = v_old + v_b1 + v_b2;

  return abs(f_bn * v_old / v_bn - f_bp * v_old / v_bp) +
         abs(f_b1 - f_bn * v_b1 / v_bn ) +
         abs(f_b2 - f_bn * v_b2 / v_bn );
}

/*
 * Apply a sibling sibling merge to the histogram
 */
static void performSiblingSiblingMerge(
    st_head_t* head, st_hole_t* parent, st_hole_t* c1, st_hole_t* c2) {
  // It is easier for us if c1 < c2:
  if (c2 < c1) {
    st_hole_t* tmp = c2;
    c2 = c1;
    c1 = tmp;
  }
  // We keep track of the relative positions of c1 and c2 in the parent's
  // children array, as this array will potentially be modified.
  int pos1 = c1 - parent->children;
  int pos2 = c2 - parent->children;
  // We also keep a copy of c1 and c2, as this makes it easier for us to release
  // resources later.
  st_hole_t c1_c = *c1;
  st_hole_t c2_c = *c2;

  kde_float_t f_b1 = c1->tuples;
  kde_float_t f_b2 = c2->tuples;
  kde_float_t f_bp = parent->tuples;
  kde_float_t v_bp = v(head, parent);

  // Initialize a new bucket as the bounding box of both c1 and c2.
  st_hole_t bn;
  initializeNewSthole(&bn, head);
  boundingBox(head, c1, &bn);
  boundingBox(head, c2, &bn);
  // Now increase the size of the bounding box until there are no remaining
  // partial intersections with any children.
  while (true) {
    int changed = 0;
    int i = 0;
    for (; i < parent->nr_children; i++) {
      st_hole_t* child = parent->children + i;
      if (child == c1 || child == c2) continue; // Skip to-be-merged children.
      // Now check whether bn intersects with this child. If it does, we have
      // to grow bn until this intersection disappears.
      intersection_t type = getIntersectionType(head, child, &bn);
      if (type == PARTIAL) {
        boundingBox(head, child, &bn);
        changed = 1;
      }
    }
    if(! changed) break;  // No more partial intersections, we are done.
  }
  // Capture the p-c2 merge corner case:
  if (getIntersectionType(head, parent, &bn) == EQUALITY) {
    // PerformParentChild will change the underlying memory of the children.
    performParentChildMerge(head, parent, parent->children + pos2);
    performParentChildMerge(head, parent, parent->children + pos1);
    releaseResources(&bn);
    return;
  }
  // Remove c1 and c2 from the parent.
  unregisterChild(head, parent, pos2);
  unregisterChild(head, parent, pos1);
  // And move all enclosed children from the parent to the new bounding box.
  int i = parent->nr_children - 1;
  for (; i >=0 ; i--) {
    intersection_t type = getIntersectionType(head, parent->children+i, &bn);
    if (type == FULL21) {
      registerChild(head, &bn, parent->children+i);
      unregisterChild(head, parent, i);
    }
  }

  //v(head,&bn) = vBox(head) - vBox(participants)
  kde_float_t v_old = fmax(v(head,&bn) - vBox(head,&c1_c) - vBox(head,&c2_c),0);
  kde_float_t f_bn = fmax(f_b1 + f_b2 + f_bp * v_old / v_bp, 0);

  bn.tuples = f_bn;
  parent->tuples = parent->tuples * (1 - v_old/v_bp);
    
  for(i = 0; i < c2_c.nr_children; i++){
    registerChild(head, &bn, c2_c.children + i);
  }

  for(i = 0; i < c1_c.nr_children; i++){
    registerChild(head, &bn, c1_c.children + i);
  }

  releaseResources(&c1_c);
  releaseResources(&c2_c);

  registerChild(head, parent, &bn);
  head->holes--;;
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
static void _getSmallestMerge(
    st_head_t* head, st_hole_t* parent, st_hole_t* hole, merge_t* best_merge) {
  kde_float_t test = 0.0;
  
  // First, we recurse to all of our children to check for their best merge.
  int i = 0;
  for (; i < hole->nr_children; i++) {
    _getSmallestMerge(head, hole, hole->children + i, best_merge);
  }

  // Check the merge with our parent.
  kde_float_t merge_cost;
  if (parent != NULL) {
    merge_cost = parentChildMergeCost(head, parent, hole);
    if (merge_cost < best_merge->penalty) {
      best_merge->penalty = merge_cost;
      best_merge->parent = parent;
      best_merge->child1 = hole;
      best_merge->child2 = NULL;
    }
  }

  // Now check the merge costs between our children.
  // First, check whether we need to build or update the cache.
  if (hole->nr_children && hole->children_merge_cost == NULL) {
    hole->children_merge_cost = malloc(
        sizeof(kde_float_t) * hole->nr_children * hole->nr_children);
    for (i = 0; i < hole->nr_children; ++i) {
      int j = i + 1;
      for (; j < hole->nr_children; ++j) {
        hole->children_merge_cost[i * hole->nr_children + j] =
            siblingSiblingMergeCost(
                head, hole, hole->children + i, hole->children + j);
      }
    }
  } else if (hole->children_merge_cost_cache_dirty) {
    for (i = 0; i < hole->nr_children; ++i) {
      int j = i + 1;
      for (; j < hole->nr_children; ++j) {
        if (hole->children_merge_cost[i * hole->nr_children + j] < 0) {
          hole->children_merge_cost[i * hole->nr_children + j] =
              siblingSiblingMergeCost(
                  head, hole, hole->children + i, hole->children + j);
        }
      }
    }
    hole->children_merge_cost_cache_dirty = 0;
  }

  for (i = 0; i < hole->nr_children; ++i) {
    int j = i + 1;
    for (; j < hole->nr_children; ++j) {
      kde_float_t merge_cost =
          hole->children_merge_cost[i * hole->nr_children + j];
      if (merge_cost < best_merge->penalty) {
        best_merge->penalty = merge_cost;
        best_merge->parent = hole;
        best_merge->child1 = hole->children + i;
        best_merge->child2 = hole->children + j;
      }
    }
  }
}

/**
 * Dumps the complete histogram to stdout
 */ 
static void printTree(st_head_t* head){
  fprintf(stderr, "Dimensions: %u\n", head->dimensions);
  fprintf(stderr, "Max #holes: %i\n", head->max_holes);
  fprintf(stderr, "Current #holes: %i\n", head->holes);
  _printTree(head, &(head->root), 0);
}

/**
 * Performs min penalty merges until we are in our memory boundaries again.
 */ 
static void mergeHoles(st_head_t* head) {
  while (head->holes > head->max_holes) {
    //fprintf(stderr,"Holes: %i > %i\n",head->holes, head->max_holes);
    merge_t bestMerge;
    bestMerge.penalty = INFINITY;
    // Get the best possible merge in the tree
    _getSmallestMerge(head, NULL, &(head->root), &bestMerge);
    // See, what kind of merge it is and perform it
    if (bestMerge.child2 != NULL) {
      //fprintf(stderr,"Sibling-sibling merge!\n");
      performSiblingSiblingMerge(
          head, bestMerge.parent, bestMerge.child1, bestMerge.child2);
    } else {
      //fprintf(stderr,"Parent-child merge!\n");
      performParentChildMerge(head, bestMerge.parent, bestMerge.child1);
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
  rc = _propagateTuple(head, &(head->root), tuple);
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
  kde_float_t ivol;
  
  //Can we answer this query?
  int i = 0;
  for (; i < request->range_count; ++i) {
    request_columns |= 0x1 << request->ranges[i].colno;
  }

  //We do not allow queries missing restrictions on a variable
  if ((head->columns | request_columns) != head->columns || request->range_count != head->dimensions) return 0;

  //Bring the request in a nicer form and store it
  setLastQuery(head,request);
  
  //"If the current query q extends the beyond the boundaries of the root bucket
  //we expand the root bucket so that it covers q." Section 4.2, p. 8
  //We usually assume closed intervals for the queries, but stholes bounds are half open intervalls (5.2).
  //We add a machine epsilon to the bound, just in case
  for(i = 0; i < head->dimensions; i++) {
    if (head->root.bounds[2*i] > head->last_query.bounds[2*i]) {
      head->root.bounds[2*i] = head->last_query.bounds[2*i];
      // Invalidate the cached volume.
      head->root.v = -1.0f;
      head->root.v_box = -1.0f;
    }
    if (head->root.bounds[2*i+1] < head->last_query.bounds[2*i+1]) {
      head->root.bounds[2*i+1] = head->last_query.bounds[2*i+1]; //+ abs(head->last_query.bounds[2*i+1]) * head->epsilon;
      head->root.v = -1.0f;
      head->root.v_box = -1.0f;
    }
  }

  
  *selectivity = _est(head, &(head->root), &ivol);
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
int stholes_est(
    Oid rel, const ocl_estimator_request_t* request, Selectivity* selectivity) {
  if (current == NULL) return 0;
  if (rel == current->table) {
    resetAllCounters(&(current->root));
    int rc = est(current, request,selectivity);
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
  
  if (((SeqScanState*) node)->ss_currentRelation->rd_id == current->table &&
      current->process_feedback) {
    float8 qual_tuples =
        (float8)(node->instrument->tuplecount + node->instrument->ntuples) /
        (node->instrument->nloops + 1);
    float8 all_tuples =
        (float8)(node->instrument->tuplecount + node->instrument->nfiltered2 +
                 node->instrument->nfiltered1 + node->instrument->ntuples) /
        (node->instrument->nloops+1);
   
    drillHoles(current);
    mergeHoles(current);
    
    ocl_reportErrorToLogFile(
        ((SeqScanState*) node)->ss_currentRelation->rd_id,
        qual_tuples / all_tuples,
        current->last_selectivity/all_tuples,
        all_tuples);
    
    current->process_feedback = 0;
  }
  
}
