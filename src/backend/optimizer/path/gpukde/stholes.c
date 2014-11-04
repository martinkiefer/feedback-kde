#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"

#include "optimizer/path/gpukde/stholes_estimator_api.h"
#include "ocl_estimator.h"
#include <executor/tuptable.h>
#include <float.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "nodes/plannodes.h"
#include <nodes/execnodes.h>
#include "executor/instrument.h"
#include "executor/tuptable.h"

struct st_hole;

typedef struct merge {
  struct st_hole* merge_partner;
  kde_float_t penalty;
} merge_t;

/**
 * An sthole instance. 
 * Contains the number of tuples it contains, its bounds and its children
 */ 
typedef struct st_hole {
  kde_float_t tuples;
  // Children of this hole.
  int nr_children;
  int child_capacity;
  struct st_hole** children;
  struct st_hole* parent;
  // Boundaries of this hole.
  kde_float_t* bounds;
  // Working counter for the statistics step.
  kde_float_t counter;
  // Cache for volume computations.
  kde_float_t v;
  kde_float_t v_box;
  // Caches the best merge for this hole.
  merge_t merge_cache;
} st_hole_t;

/**
 * The head of an stholes histogram. 
 * Contains additional meta information
 */ 
typedef struct st_head {
  // Root hole.
  st_hole_t* root;
  // Meta information.
  int holes;
  int max_holes;
  unsigned int dimensions;
  Oid table;
  int32 columns;
  unsigned int* column_order;
  // Information for updating the structure.
  st_hole_t* last_query;
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
bool stholes_enabled() {
  return stholes_enable;
}  

static void _printTree(st_head_t* head, st_hole_t* hole, int depth);

/**
 * Create a new empty st hole
 */
static st_hole_t* initializeNewSTHole(const st_head_t* head) {
  st_hole_t* hole = calloc(1, sizeof(st_hole_t));
  // Now allocate the bounds and initialize them with +/- infinity.
  hole->bounds = (kde_float_t*) malloc(sizeof(kde_float_t)*head->dimensions*2);
  int i = 0;
  for (; i<head->dimensions ; i++) {
    hole->bounds[i*2] = INFINITY;
    hole->bounds[i*2+1] = -INFINITY;
  }
  // Initialize the cached computations.
  hole->v = -1.0f;
  hole->v_box = -1.0f;
  hole->merge_cache.penalty = INFINITY;
  // Return the result.
  return hole;
}

/**
 * Create a new head bucket for the histogram
 */
static st_head_t* createNewHistogram(
    Oid table, AttrNumber* attributes, unsigned int dimensions) {
  st_head_t* head = (st_head_t*) calloc(1,sizeof(st_head_t));
  
  head->dimensions = dimensions;
  head->table = table;
  head->process_feedback = 0;
  head->max_holes = stholes_hole_limit;
  head->holes = 1;
  head->column_order = calloc(1, 32 * sizeof(unsigned int));
  
  head->root = initializeNewSTHole(head);
  head->last_query = initializeNewSTHole(head);

  if (sizeof(kde_float_t) == sizeof(double)) {
    head->epsilon = DBL_EPSILON;
  } else {
    head->epsilon = FLT_EPSILON;
  }

  int i = 0;
  for (; i<dimensions; ++i) {
     head->columns |= 0x1 << attributes[i];
     head->column_order[attributes[i]] = i;
  }

  return head;
}

// Helper function to reset merge cache references referring to a given bucket.
static void resetMergeCache(st_hole_t* hole) {
  unsigned int i;
  // First, we check whether any of our siblings have selected us
  // for a sibling-sibling merge.
  if (hole->parent) {
    for (i=0; i<hole->parent->nr_children; ++i) {
      if (hole->parent->children[i]->merge_cache.merge_partner == hole) {
        // Reset this child.
        hole->parent->children[i]->merge_cache.penalty = INFINITY;
      }
    }
  }
  // Next, we need to check whether any of our children have selected us for a
  // parent-child merge.
  for (i=0; i<hole->nr_children; ++i) {
    if (hole->children[i]->merge_cache.merge_partner == hole) {
      hole->children[i]->merge_cache.penalty = INFINITY;
    }
  }
  // Finally, we invalidate our own merge cache.
  hole->merge_cache.penalty = INFINITY;
}

/**
 * Release the resources of an sthole
 */
static void releaseResources(st_hole_t* hole) {
  free(hole->bounds);
  free(hole->children);
  free(hole);
}

static void _destroyHistogram(st_hole_t* hole) {
  int i = 0;
  for (; i < hole->nr_children; i++) {
    _destroyHistogram(hole->children[i]);
  }
  releaseResources(hole);
}

static void destroyHistogram(st_head_t* head) {
  _destroyHistogram(head->root);
  releaseResources(head->last_query);
  free(head);
}


// Convert the last query to an sthole.
// We can then simply use our standard functions for vBox and v for it.
static void setLastQuery(
    st_head_t* head, const ocl_estimator_request_t* request) {
  int i = 0;
  for (; i < request->range_count; i++) {
    // Add tiny little epsilons, if necessary, to account for the [) buckets.
    if (request->ranges[i].lower_included) {
      head->last_query->bounds[head->column_order[request->ranges[i].colno]*2] =
          request->ranges[i].lower_bound;
    } else {
      head->last_query->bounds[head->column_order[request->ranges[i].colno]*2] =
          request->ranges[i].lower_bound +
          fabs(request->ranges[i].lower_bound) * head->epsilon;
    }
    if (request->ranges[i].upper_included) {
      head->last_query->bounds[head->column_order[request->ranges[i].colno]*2+1] =
          request->ranges[i].upper_bound +
          fabs(request->ranges[i].upper_bound) * head->epsilon;
    } else {
      head->last_query->bounds[head->column_order[request->ranges[i].colno]*2+1] =
          request->ranges[i].upper_bound;
    }
  }
  // Invalidate the volume cache.
  head->last_query->v = -1.0f;
  head->last_query->v_box = -1.0f;
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
    for (; i < hole->nr_children; i++) {
      v -= vBox(head, hole->children[i]);
    }
    Assert(v >= -1.0); // This usually means that something went terribly wrong.
    hole->v = fmax(0.0, v);
  }
  return hole->v;
}

/**
 * Add a new child to the given parent.
 */
static void registerChild(
    const st_head_t* head, st_hole_t* parent, st_hole_t* child) {
  parent->nr_children++;
  if (parent->child_capacity < parent->nr_children) {
    // Increase the child capacity by 10.
    parent->child_capacity += 10;
    parent->children = realloc(
        parent->children, parent->child_capacity * sizeof(st_hole_t*));
  }
  parent->children[parent->nr_children - 1] = child;
  child->parent = parent;
  // We changed the children, update the volume cache.
  if (parent->v > -1.0f) {
    parent->v -= vBox(head, child);
    parent->v = fmax(parent->v, 0.0f);
  }
}

/**
 * Remove the child at position pos in bucket parent
 */
static void unregisterChild(
    st_head_t* head, st_hole_t* parent, st_hole_t* child) {
  int i=0;
  for (; i<parent->nr_children; ++i) {
    if (parent->children[i] == child && i != parent->nr_children - 1) {
      parent->children[i] = parent->children[parent->nr_children - 1];
    }
  }
  parent->nr_children--;
  // Update the volume cache.
  if (parent->v > -1.0f) {
    parent->v += vBox(head, child);
  }
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
        head->last_query->bounds[2*i], hole->bounds[2*i]);
    target_hole->bounds[2*i+1] = fmin(
        head->last_query->bounds[2*i+1], hole->bounds[2*i+1]);
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
  for (; i < head->dimensions; i++) {
    //Case 1: We have no intersection with this hole
    //If this does not intersect with one of the intervals of the box, we have nothing to do.
    //Neither have our children.
    if (hole1->bounds[2*i+1] <= hole2->bounds[2*i] ||
        hole1->bounds[2*i] >= hole2->bounds[2*i+1]) {
      return NONE;
    }

    if (!(hole2->bounds[2*i] >= hole1->bounds[2*i] &&
          hole2->bounds[2*i+1] <= hole1->bounds[2*i+1])) {
      enclosed12 = 0;
    }

    if (!(hole1->bounds[2*i] >= hole2->bounds[2*i] &&
          hole1->bounds[2*i+1] <= hole2->bounds[2*i+1])) {
      enclosed21 = 0;
    }
  }
  if (enclosed21 && enclosed12) {
    return EQUALITY;
  } else if (enclosed21) {
    return FULL21;
  } else if (enclosed12) {
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
  int i = 0;
  for (; i < hole->nr_children; i++) {
    int j;
    for (j = i+1; j < hole->nr_children; j++) {
      if (getIntersectionType(
           head, hole->children[i], hole->children[j]) != NONE) {
        fprintf(stderr, "Intersection between child %i and %i is %i\n", i, j,
                getIntersectionType(head,hole->children[i], hole->children[j]));
        return 0;
      }
    }
  }
  return 1;
}

/**
 * Aggregate estimated tuples recursively
 */
static kde_float_t _est(const st_head_t* head, st_hole_t* hole,
    kde_float_t* intersection_vol) {
  kde_float_t est = 0.0;
  *intersection_vol = 0;
  
  // If we don't intersect with the query, the estimate is zero.
  if (getIntersectionType(head, hole, head->last_query) == NONE) {
    return est;
  }
  
  //  Calculate v(q b)
  st_hole_t* q_i_b = initializeNewSTHole(head);
  intersectWithLastQuery(head, hole, q_i_b);
  *intersection_vol = vBox(head, q_i_b);
  
  kde_float_t v_q_i_b = *intersection_vol;
  
  // Recurse over all children to sum up estimates and remove child volumes.
  int i = 0;
  for (; i < hole->nr_children; i++) {
    kde_float_t child_intersection;
    est += _est(head, hole->children[i], &child_intersection);
    v_q_i_b -= child_intersection;
  }
  // Ensure that v_q_i_b is capped by zero.
  v_q_i_b = fmax(0.0, v_q_i_b);
  
  // If the hole is overly filled, we might run into numerical issues here
  // that lead to empty estimates. Therefore we only add up the estimates
  // for holes that we consider safe.
  kde_float_t vh = v(head, hole);
  if (vh >= fabs(head->epsilon * vBox(head, hole))) {
    est += hole->tuples * (v_q_i_b / vh); 
  }
  
  releaseResources(q_i_b);
  
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
    if (_propagateTuple(head, hole->children[i], tuple) == 1) return 1;
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
    resetAllCounters(hole->children[i]);
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
    st_head_t* head, st_hole_t* intersection, st_hole_t* hole) {
  int max_dim = -1;
  kde_float_t max_vol = -INFINITY;
  int i = 0;
  for (; i < head->dimensions; i++) {
    kde_float_t vol = getReducedVolume(head, intersection, hole, i);
    if (vol > max_vol) {
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
  int i, j, old_hole_size = 0;
  intersection_t type;
  kde_float_t v_qib;

  if (getIntersectionType(head, head->last_query, hole) == NONE) return;
  st_hole_t* candidate = initializeNewSTHole(head);
  st_hole_t* tmp = initializeNewSTHole(head);
  // Get the intersection with the last query for this hole.
  intersectWithLastQuery(head, hole, candidate);
  v_qib = vBox(head, candidate); //*Will be adjusted to the correct value later
  type = getIntersectionType(head, hole, candidate);
  
  // Shrink the candidate hole.
  switch(type) {
    case NONE:
      return;
    
    // Case 2: We have complete intersection with this hole. Update stats.
    case EQUALITY:
      hole->tuples = hole->counter;
      goto nohole;
      
    
    case FULL12:
      //We will use the tmp
      for (i = 0; i < hole->nr_children; i++) {
        if (getIntersectionType(
            head, head->last_query, hole->children[i]) != NONE) {
          intersectWithLastQuery(head, hole->children[i], tmp);
          v_qib -= vBox(head, tmp);
        }
      } 
      releaseResources(tmp);
      // Shrink the candidate until it does not intersect any children.
      while (true) {
        int changed = 0;
        for (i = 0; i < hole->nr_children; i++) {
          // Full intersections:
          intersection_t type = getIntersectionType(
              head, hole->children[i], candidate);
          // Full and no intersection are no problem:
          if (type == FULL12) {
            goto nohole;
          } else if (type == NONE) {
            continue;
          } else if (type == EQUALITY) {
            //The child can handle this case on its own.
            goto nohole;
          } else if (type == FULL21) {
            continue; //The child is completely located inside the intersection.
          } else {
            unsigned int max_dim =
                minReducedVolumeDimension(head, candidate, hole->children[i]);
            shrink(head, candidate, hole->children[i], max_dim);
            changed = 1;
          }
        }
        if (!changed) break;
      }
      break;
      
    case PARTIAL:
    case FULL21:
      //The construction of the intersection forbids this case
      Assert(0);
      break;
  }
  
  //This is a shitty corner case, that can occur.
  if (vBox(head, candidate) <= fabs(head->epsilon*vBox(head, candidate))) {
    goto nohole;
  }
    
  // See if we need to transfer children to the new hole
  // We run the loop backwards, because unregister child substitutes
  // from the back.
  for (i=hole->nr_children-1; i >=0 ; i--) {
    intersection_t type = getIntersectionType(
        head, candidate, hole->children[i]);
    if (type == FULL12) {
      registerChild(head, candidate, hole->children[i]);
      unregisterChild(head, hole, hole->children[i]);
    } else if (type == EQUALITY) {
      Assert(0);
    }
    Assert(type != PARTIAL);
  }
  
  candidate->tuples = hole->counter * (v(head, candidate)/v_qib);
  candidate->counter = hole->counter * (v(head, candidate)/v_qib);

  // We will now adjust the merge cache of the children that fall into the new
  // candidate hole. In particular, all caches of candidate's children that
  // refer to remaining children of the hole that will be drilled into or to
  // the hole itself have to be reset.
  for (i = 0; i < candidate->nr_children; ++i) {
    st_hole_t* child = candidate->children[i];
    if (child->merge_cache.penalty == INFINITY) continue;
    if (child->merge_cache.merge_partner == hole) {
      // Invalid, since the parent of child is now bn.
      child->merge_cache.penalty = INFINITY;
    } else {
      // Check if the child's cache refers to any of parent's children.
      for (j = 0; j < hole->nr_children; ++j) {
        if (child->merge_cache.merge_partner == hole->children[j]) {
          child->merge_cache.penalty = INFINITY;
          break;
        }
      }
    }
  }
  // Now adjust the merge cache of the remaining children of the hole that will
  // be drilled into. In particular, all caches of the hole's children that
  // refer to a child of the candidate hole need to be reset.
  for (i = 0; i < hole->nr_children; ++i) {
    st_hole_t* child = hole->children[i];
    if (child->merge_cache.penalty == INFINITY) continue;
    for (j = 0; j < candidate->nr_children; ++j) {
      if (child->merge_cache.merge_partner == candidate->children[j]) {
        child->merge_cache.penalty = INFINITY;
        break;
      }
    }
  }

  // Finally, register the candidate hole.
  registerChild(head, hole, candidate);
  
  Assert(_disjunctivenessTest(head, candidate));
  Assert(_disjunctivenessTest(head, hole));
  
  hole->tuples = fmax(hole->tuples - candidate->tuples, 0.0);
  
  // Does this bucket still carry information?
  // If not, migrate all children to the parent. Of course, the root bucket can't be removed.
  if (parent != NULL && v(head, hole) <= fabs(head->epsilon*vBox(head,hole))) {
    resetMergeCache(hole);  // Invalidate all cache entries to the hole.
    unregisterChild(head, parent, hole);
    
    int old_parent_size = parent->nr_children;
    for (i = 0; i < hole->nr_children; i++) {
      registerChild(head, parent, hole->children[i]);
    }
    
    for (i=old_parent_size; i < hole->nr_children; i++) {
      _drillHoles(head, parent, parent->children[i]);
    }

    head->holes--;

    releaseResources(hole);
    
    Assert(_disjunctivenessTest(head,parent));
    return;
  }
  head->holes++;
  
  // Tell the children about the new query.
  old_hole_size = hole->nr_children;
  for (i=0; i < old_hole_size; i++) {
    _drillHoles(head, hole, hole->children[i]);
  } 
  return;
  
nohole:
  releaseResources(candidate);
  old_hole_size = hole->nr_children;
  for (i=0; i < old_hole_size; i++) {
    _drillHoles(head, hole, hole->children[i]);
  }
  return;
}


static void drillHoles(st_head_t* head) {
  _drillHoles(head, NULL, head->root);
}

/**
 * Calculate the parent child merge penalty for a given pair of buckets
 */
static kde_float_t parentChildMergeCost(
    const st_head_t* head, st_hole_t* parent, st_hole_t* child) {
  if (parent == NULL) return INFINITY;
    
  kde_float_t fbp = parent->tuples;
  kde_float_t fbc = child->tuples;
  kde_float_t fbn = fbc + fbp;
  kde_float_t vbp = v(head, parent);
  kde_float_t vbc = v(head, child);
  kde_float_t vbn = vbp + vbc;
  
  return fabs(fbp - fbn * vbp / vbn) + fabs(fbc - fbn * vbc / vbn);
}

/**
 * Calculate the penalty for a parent double child merge (Corner case
 * of a sibling sibling merge)
 */
static kde_float_t parentChildChildMergeCost(
    const st_head_t* head, st_hole_t* parent, st_hole_t* c1, st_hole_t* c2) {
  if (parent == NULL) return INFINITY;
    
  kde_float_t fbp = parent->tuples;
  kde_float_t fbc1 = c1->tuples;
  kde_float_t fbc2 = c2->tuples;
  kde_float_t fbn = fbc1 + fbc2 + fbp;
  kde_float_t vbp = v(head, parent);
  kde_float_t vbc1 = v(head, c1);
  kde_float_t vbc2 = v(head, c2);
  kde_float_t vbn = vbp + vbc1 + vbc2;
  
  return fabs(fbp - fbn * vbp / vbn) +
         fabs(fbc1 - fbn * vbc1 / vbn) +
         fabs(fbc2 - fbn * vbc2 / vbn);
}

/**
 * Apply a parent child merge to the histogram
 */
static void performParentChildMerge(
    st_head_t* head, st_hole_t* parent, st_hole_t* child) {
  parent->tuples += child->tuples;
  
  // We need to reset all mergecache entries refering to the to-be-merged child,
  // as those will become invalid.
  resetMergeCache(child);

  // Now we migrate all grandchildren to the parent.
  int i = 0;
  for (; i < child->nr_children; i++) {
    registerChild(head, parent, child->children[i]);
  }
  
  unregisterChild(head, parent, child);
  releaseResources(child);
 
  head->holes--;
}

/**
 * Calculate the cost for an sibling sibling merge
 */
static kde_float_t siblingSiblingMergeCost(
    const st_head_t* head, st_hole_t* parent, st_hole_t* c1, st_hole_t* c2) {
  int i;
  st_hole_t* bn = initializeNewSTHole(head);

  kde_float_t f_b1 = c1->tuples;
  kde_float_t f_b2 = c2->tuples;
  kde_float_t f_bp = parent->tuples;
  kde_float_t v_bp = v(head, parent);
  kde_float_t v_b1 = v(head, c1);
  kde_float_t v_b2 = v(head, c2);

  boundingBox(head, c1, bn);
  boundingBox(head, c2, bn);

  // Progressively increase the size of the bounding box until we have
  // no partial intersections.
  kde_float_t v_I = 0;
  while (true) {
    bool first_pass = true;
    int changed = 0;
    for (i = 0; i < parent->nr_children; i++) {
      st_hole_t* child = parent->children[i];
      if (child == c1 || child == c2) continue; // Skip to-be-merged children.
      // Now check whether bn intersects with this child. If it does, we have
      // to grow bn until this intersection disappears.
      intersection_t type = getIntersectionType(head, child, bn);
      if (type == PARTIAL) {
        v_I += vBox(head, child);
        boundingBox(head, child, bn);
        changed = 1;
      } else if (first_pass && type == FULL21) {
        // On the first pass, we also need to take holes that our new hole
        // already fully contains into account to compute the displaced volume.
        v_I += vBox(head, child);
      }
      first_pass = false;
    }
    if (!changed) break;
  }

  // Check if this is a special case:
  if (getIntersectionType(head, parent, bn) == EQUALITY) {
    releaseResources(bn);
    return parentChildChildMergeCost(head, parent, c1, c2);
  }

  // Compute the displaced volume by the new box.
  kde_float_t v_old = v(head, bn) - vBox(head, c1) - vBox(head, c2) - v_I;
  releaseResources(bn);
  kde_float_t f_bn = f_b1 + f_b2 + f_bp * v_old / v_bp;

  // Caution: there is an error in the full paper saying this is the value for
  // v_p. It obviously is v_bn.
  kde_float_t v_bn = v_old + v_b1 + v_b2;

  return fabs(f_bn * v_old / v_bn - f_bp * v_old / v_bp) +
         fabs(f_b1 - f_bn * v_b1 / v_bn ) +
         fabs(f_b2 - f_bn * v_b2 / v_bn );
}

/*
 * Apply a sibling-sibling merge to the histogram.
 */
static void performSiblingSiblingMerge(
    st_head_t* head, st_hole_t* parent, st_hole_t* c1, st_hole_t* c2) {
  int i, j;

  kde_float_t f_b1 = c1->tuples;
  kde_float_t f_b2 = c2->tuples;
  kde_float_t f_bp = parent->tuples;
  kde_float_t v_bp = v(head, parent);

  // Before we do anything, reset the merge cache of c1 and c2.
  resetMergeCache(c1);
  resetMergeCache(c2);

  // Initialize a new bucket as the bounding box of both c1 and c2.
  st_hole_t* bn = initializeNewSTHole(head);
  boundingBox(head, c1, bn);
  boundingBox(head, c2, bn);

  // Now increase the size of the bounding box until there are no remaining
  // partial intersections with any children.
  while (true) {
    int changed = 0;
    for (i = 0; i < parent->nr_children; i++) {
      st_hole_t* child = parent->children[i];
      if (child == c1 || child == c2) continue; // Skip to-be-merged children.
      // Now check whether bn intersects with this child. If it does, we have
      // to grow bn until this intersection disappears.
      intersection_t type = getIntersectionType(head, child, bn);
      if (type == PARTIAL) {
        boundingBox(head, child, bn);
        changed = 1;
      }
    }
    if (!changed) break;  // No more partial intersections, we are done.
  }
  // Capture the p-c2 merge corner case:
  if (getIntersectionType(head, parent, bn) == EQUALITY) {
    performParentChildMerge(head, parent, c1);
    performParentChildMerge(head, parent, c2);
    releaseResources(bn);
    return;
  }

  // Now move all fully enclosed children (besides c1 and c2) from the parent
  // to the new hole.
  for (i=parent->nr_children - 1; i>=0 ; i--) {
    if (parent->children[i] == c1 || parent->children[i] == c2) continue;
    intersection_t type = getIntersectionType(head, parent->children[i], bn);
    if (type == FULL21) {
      registerChild(head, bn, parent->children[i]);
      unregisterChild(head, parent, parent->children[i]);
    }
  }

  // Update the tuple counts for both the new bucket and the parent.
  kde_float_t v_old = fmax(v(head, bn) - vBox(head, c1) - vBox(head, c2), 0);
  kde_float_t f_bn = fmax(f_b1 + f_b2 + f_bp * v_old / v_bp, 0);
  bn->tuples = f_bn;
  parent->tuples = parent->tuples * (1 - v_old/v_bp);

  // Move the children of c1 and c2 to the new bucket.
  for (i = 0; i < c1->nr_children; i++) {
    registerChild(head, bn, c1->children[i]);
  }
  for (i = 0; i < c2->nr_children; i++) {
    registerChild(head, bn, c2->children[i]);
  }

  // And unregister and delete c1 and c2.
  unregisterChild(head, parent, c1);
  unregisterChild(head, parent, c2);
  releaseResources(c1);
  releaseResources(c2);

  // We will now adjust the merge cache of the children of bn. In particular,
  // all caches of bn's children that refer to children of the parent or to
  // the parent itself have to be reset.
  for (i = 0; i < bn->nr_children; ++i) {
    st_hole_t* child = bn->children[i];
    if (child->merge_cache.penalty == INFINITY) continue;
    if (child->merge_cache.merge_partner == parent) {
      // Invalid, since the parent of child is now bn.
      child->merge_cache.penalty = INFINITY;
    } else {
      // Check if the child's cache refers to any of parent's children.
      for (j = 0; j < parent->nr_children; ++j) {
        if (child->merge_cache.merge_partner == parent->children[j]) {
          child->merge_cache.penalty = INFINITY;
          break;
        }
      }
    }
  }
  // Now adjust the merge cache of children of parent. In particular, all
  // caches of parent's children that refer to a child of bn need to be reset.
  for (i = 0; i < parent->nr_children; ++i) {
    st_hole_t* child = parent->children[i];
    if (child->merge_cache.penalty == INFINITY) continue;
    for (j = 0; j < bn->nr_children; ++j) {
      if (child->merge_cache.merge_partner == bn->children[j]) {
        child->merge_cache.penalty = INFINITY;
        break;
      }
    }
  }

  // Finally, register bn as a new child of the parent.
  registerChild(head, parent, bn);
  head->holes--;
}

/**
 * Find a min cost merge in the tree
 */ 
static void _getSmallestMerge(
    st_head_t* head, st_hole_t* parent, st_hole_t* hole, merge_t* best_merge) {
  Assert(hole->parent == parent); // Ensure the structural integrity is correct.

  // First, we recurse to all of our children to check for their best merge.
  int i = 0;
  for (; i < hole->nr_children; i++) {
    _getSmallestMerge(head, hole, hole->children[i], best_merge);
  }

  // Check if we need to recompute the merge cache.
  if (hole->merge_cache.penalty == INFINITY) {
    // We need to recompute the cheapest merge.
    // First, we check the merge against our parent.
    if (parent != NULL) {
      // First, we check how expensive a merge with our parent would be.
      hole->merge_cache.penalty = parentChildMergeCost(head, parent, hole);
      hole->merge_cache.merge_partner = parent;
      // Next, we check the merge costs with our siblings:
      for (i = 0; i < hole->parent->nr_children; ++i) {
        st_hole_t* sibling = hole->parent->children[i];
        if (sibling != hole) {
          kde_float_t merge_cost = siblingSiblingMergeCost(
              head, hole->parent, hole, sibling);
          if (merge_cost < hole->merge_cache.penalty) {
            hole->merge_cache.penalty = merge_cost;
            hole->merge_cache.merge_partner = sibling;
          }
        }
      }
    }
  }

  // Ok, we have a valid merge cache. Check if we are the currently cheapest
  // one.
  if (hole->merge_cache.penalty < best_merge->penalty) {
    best_merge->penalty = hole->merge_cache.penalty;
    best_merge->merge_partner = hole;
  }
}

/**
 * Dumps the complete histogram to stdout
 */ 
static void printTree(st_head_t* head) {
  fprintf(stderr, "Dimensions: %u\n", head->dimensions);
  fprintf(stderr, "Max #holes: %i\n", head->max_holes);
  fprintf(stderr, "Current #holes: %i\n", head->holes);
  _printTree(head, head->root, 0);
}

/**
 * Performs min penalty merges until we are in our memory boundaries again.
 */ 
static void mergeHoles(st_head_t* head) {
  while (head->holes > head->max_holes) {
    merge_t bestMerge;
    bestMerge.penalty = INFINITY;
    // Get the best possible merge in the tree
    _getSmallestMerge(head, NULL, head->root, &bestMerge);
    // See, what kind of merge it is and run it:
    st_hole_t* merge_partner_1 = bestMerge.merge_partner;
    st_hole_t* merge_partner_2 = merge_partner_1->merge_cache.merge_partner;
    if (merge_partner_2 == merge_partner_1->parent) {
      // Parent-Child merge, with merge_partner_1 being the child.
      performParentChildMerge(head, merge_partner_2, merge_partner_1);
    } else {
      // Sibling-Sibling merge between the two nodes.
      performSiblingSiblingMerge(
          head, merge_partner_1->parent, merge_partner_1, merge_partner_2);
    }
  }
}

/**
 * Is called for every qualifying tuple processed in a sequential scan
 * and then finds the correct bucket and increases the counter 
*/
static void propagateTuple(
    st_head_t* head, Relation rel, const TupleTableSlot* slot) {
  
  int i = 0;
  bool isNull;
  TupleDesc desc = slot->tts_tupleDescriptor;
  kde_float_t* tuple = (kde_float_t*) malloc(
      sizeof(kde_float_t)*head->dimensions);
  
  HeapTuple htup = slot->tts_tuple;
  Assert(htup);

  for (i = 0; i < desc->natts; i++) {
    // Skip columns that are requested but not handled by our estimator.
    if (! (head->columns & (0x1 << desc->attrs[i]->attnum))) continue;
    
    Datum datum = heap_getattr(
        htup, desc->attrs[i]->attnum ,RelationGetDescr(rel), &isNull);
    
    Assert(desc->attrs[i]->atttypid == FLOAT8OID || 
           desc->attrs[i]->atttypid == FLOAT4OID);
    
    
    if (desc->attrs[i]->atttypid == FLOAT8OID) {
      tuple[head->column_order[desc->attrs[i]->attnum]] =
          (kde_float_t) DatumGetFloat8(datum);
    }
    else {
      tuple[head->column_order[desc->attrs[i]->attnum]] =
          (kde_float_t) DatumGetFloat4(datum);
    }
    
  } 
  
  //Very well, by now we should have a nice tuple. Lets do some traversing.
  int rc = _propagateTuple(head, head->root, tuple);
  Assert(rc); // We should never encounter a tuple that isn't claimed.
}

/**
 * API method to propagate a tuple from the result stream.
 */
void stholes_propagateTuple(Relation rel, const TupleTableSlot* slot) {
  if (current != NULL && rel->rd_id == current->table) {
    propagateTuple(current, rel, slot);
  }
}

/**
 * API method to create a new histogram and remove the old one.
 */
void stholes_addhistogram(
    Oid table, AttrNumber* attributes, unsigned int dimensions) {
  if (current != NULL) destroyHistogram(current);
  current = createNewHistogram(table,attributes,dimensions);
  fprintf(stderr, "Created a new stholes histogram\n");
  
}  

static int est(
    st_head_t* head, const ocl_estimator_request_t* request,
    Selectivity* selectivity) {
  int request_columns = 0;
  kde_float_t ivol;
  
  // Can we answer this query?
  int i = 0;
  for (; i < request->range_count; ++i) {
    request_columns |= 0x1 << request->ranges[i].colno;
  }

  // We do not allow queries missing restrictions on a variable.
  if ((head->columns | request_columns) != head->columns ||
      request->range_count != head->dimensions) {
    return 0;
  }

  // Bring the request in a nicer form and store it
  setLastQuery(head,request);

  // "If the current query q extends the beyond the boundaries of the root bucket
  // we expand the root bucket so that it covers q." Section 4.2, p. 8
  // We usually assume closed intervals for the queries, but stholes bounds are
  // half-open intervalls (5.2). We add a machine epsilon to the bound, just in case
  for (i = 0; i < head->dimensions; i++) {
    if (head->root->bounds[2*i] > head->last_query->bounds[2*i]) {
      head->root->bounds[2*i] = head->last_query->bounds[2*i];
      // Invalidate the cached volume and the merge-cache.
      head->root->v = -1.0f;
      head->root->v_box = -1.0f;
      resetMergeCache(head->root);
    }
    if (head->root->bounds[2*i+1] < head->last_query->bounds[2*i+1]) {
      head->root->bounds[2*i+1] = head->last_query->bounds[2*i+1]; //+ abs(head->last_query.bounds[2*i+1]) * head->epsilon;
      head->root->v = -1.0f;
      head->root->v_box = -1.0f;
      resetMergeCache(head->root);
    }
  }

  *selectivity = _est(head, head->root, &ivol);
  return 1;
} 


static void _printTree(st_head_t* head, st_hole_t* hole, int depth) {
  int i = 0;
  for (i=0; i < depth; i++) {
    fprintf(stderr,"\t");
  }
  
  for (i = 0; i < head->dimensions; i++) {
    fprintf(stderr,"[%f , %f] ", hole->bounds[2*i], hole->bounds[2*i+1]);
  }
  fprintf(stderr,"Counter %f",hole->counter);
  fprintf(stderr," Tuples %f",hole->tuples);
  fprintf(stderr,"\n");
  for (i = 0; i < hole->nr_children; i++) {
    _printTree(head, hole->children[i], depth+1);
  }  
}  

/**
 * API method called by the optimizer to retrieve a selectivity estimation.
 */
int stholes_est(
    Oid rel, const ocl_estimator_request_t* request, Selectivity* selectivity) {
  if (current == NULL) return 0;
  if (rel == current->table) {
    resetAllCounters(current->root);
    int rc = est(current, request, selectivity);
    current->last_selectivity = *selectivity;
    current->process_feedback = 1;
    return rc;
  }
  return 0;
}  

static void buildAndRefine(st_head_t* model) {
  // First, we need to identify candidate holes and drill them.
  drillHoles(model);
  // Then, we need to merge superfluous holes until we are within our
  // resource constraints.
  mergeHoles(model);
}

/**
 *  API method called to initiate the histogram optimization process
 */
void stholes_process_feedback(PlanState *node) {
  if (current == NULL) return;
  if (nodeTag(node) != T_SeqScanState) return;
  if (node->instrument == NULL || node->instrument->kde_rq == NULL) return;

  if (((SeqScanState*) node)->ss_currentRelation->rd_id == current->table &&
      current->process_feedback) {
    // Count the statistics.
    float8 qual_tuples =
        (float8)(node->instrument->tuplecount + node->instrument->ntuples) /
        (node->instrument->nloops + 1);
    float8 all_tuples =
        (float8)(node->instrument->tuplecount + node->instrument->nfiltered2 +
                 node->instrument->nfiltered1 + node->instrument->ntuples) /
        (node->instrument->nloops+1);
    // Update the model.
    buildAndRefine(current);
    // And report the estimation error.
    ocl_reportErrorToLogFile(
        ((SeqScanState*) node)->ss_currentRelation->rd_id,
        current->last_selectivity / all_tuples,
        qual_tuples / all_tuples,
        all_tuples);
    // We are done processing the feedback :)
    current->process_feedback = 0;
  }
}
