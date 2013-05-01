#ifndef COVER_TREE_H
#define COVER_TREE_H

#include<string.h>
#include<string>
#include<math.h>
#include<list>
#include<stdlib.h>
#include<assert.h>
#include "point.h"
#include "stack.h"

#define NUM_VECTORS 2


struct node {
  point p;
  double max_dist;  // The maximum distance to any grandchild.
  double parent_dist; // The distance to the parent.
  node* children;
  unsigned short int num_children; // The number of children.
  short int scale; // Essentially, an upper bound on the distance to any child.

  // additional info to support vector multiplication
  int point_idx;
  double distance_to_query;
  double unweighted_sum_v[NUM_VECTORS]; // unweighted sums of each vector, over the points contained at this node
  double ws[NUM_VECTORS];
  bool cutoff[NUM_VECTORS];
};

void print(int depth, node &top_node);

//construction
node new_leaf(const point &p);
node batch_create(const std::vector<point> &points, distfn distance, const double * dist_params);
//node insert(point, node *top_node); // not yet implemented
//void remove(point, node *top_node); // not yet implemented
//query
void k_nearest_neighbor(const node &tope_node, const node &query,
			v_array<v_array<point> > &results, int k, distfn distance, const double * dist_params);
void epsilon_nearest_neighbor(const node &tope_node, const node &query,
			      v_array<v_array<point> > &results, double epsilon, distfn distance, const double * dist_params);
void unequal_nearest_neighbor(const node &tope_node, const node &query,
			      v_array<v_array<point> > &results, distfn distance, const double * dist_params);

//information gathering
int height_dist(const node top_node,v_array<int> &heights);
void breadth_dist(const node top_node,v_array<int> &breadths);
void depth_dist(int top_scale, const node top_node,v_array<int> &depths);

#endif
