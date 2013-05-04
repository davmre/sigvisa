#ifndef COVER_TREE_H
#define COVER_TREE_H

#include<string.h>
#include<string>
#include<math.h>
#include<list>
#include<stdlib.h>
#include<assert.h>
#include <vector>
#include "stack.h"

#include <limits.h>
#include <values.h>
#include <stdint.h>
#include <iostream>


struct point {
  const double * p;
  unsigned int idx;
};

struct pairpoint {
  const double * pt1;
  const double * pt2;
  unsigned int idx1;
  unsigned int idx2;
};

inline pairpoint double_point(const double *p) { pairpoint pp = {p, p, 0, 0}; return pp; }

template<typename T> struct distfn {
  typedef double (*Type)(const T, const T, double, const double*);
};


double pair_distance(const point pt1, const point pt2, double BOUND_IGNORED, const double * PARAMS_IGNORED);
double dist_km(const double *p1, const double *p2);
double dist_3d_km(const point p1, const point p2, double BOUND_IGNORED, const double *scales);
double pair_dist_3d_km(const pairpoint p1, const pairpoint p2, double BOUND_IGNORED, const double *scales);
double w_se(double d, const double * variance);
typedef double (*wfn)(double, const double *);




template<class T>
class node {
 public:
  T p;
  double max_dist;  // The maximum distance to any grandchild.
  double parent_dist; // The distance to the parent.
  node<T>* children;
  unsigned short int num_children; // The number of children.
  short int scale; // Essentially, an upper bound on the distance to any child.

  // additional info to support vector multiplication
  double distance_to_query;

  unsigned int narms;
  double *unweighted_sums; // unweighted sums of each vector, over the points contained at this node
  double unweighted_sum;

  node();
  void alloc_arms(unsigned int narms);
  void free_tree();
};

template<class T>
node<T>::node() {
  /*
    WARNING: because I haven't defined a copy constructor, crazy
    things can happen with the default copy constructor where one node
    ends up pointing at the internals of another node. I should fix this.
   */

  this->unweighted_sums = &(this->unweighted_sum);
  this->narms = 1;
}

template<class T>
void node<T>::alloc_arms(unsigned int narms) {
  if (this->narms > 1) {
    delete this->unweighted_sums;
    this->unweighted_sums = NULL;
  }

  if ( narms > 1 ) {
    this->unweighted_sums = new double[narms];
  } else {
    this->unweighted_sums = &(this->unweighted_sum);
  }
  this->narms = narms;

  // recursively apply to all children
  for(unsigned int i=0; i < this->num_children; ++i) {
    this->children[i].alloc_arms(narms);
  }
}

template<class T>
void node<T>::free_tree() {
  if (this->narms > 1) {
    delete this->unweighted_sums;
  }

  for(unsigned int i=0; i < num_children; ++i) {
    delete (this->children + i);
  }
}


template <class T>
struct ds_node {
  v_array<double> dist;
  T p;
};

const double base = 1.3;
const double il2 = 1. / log(base);

inline double dist_of_scale (int s)
{

  return pow(base, s);
}

inline int get_scale(double d)
{

  return (int) ceilf(il2 * log(d));
}

inline int min(int f1, int f2)
{
  if ( f1 <= f2 )
    return f1;
  else
    return f2;
}

inline double max(double f1, double f2)
{
  if ( f1 <= f2 )
    return f2;
  else
    return f1;
}

template <class T>
node<T> new_node(const ds_node<T> &p)
{
  node<T> new_node;
  new_node.p = p.p;
  return new_node;
}

template <class T>
node<T> new_leaf(const ds_node<T> &p)
{
  node<T> new_leaf;
  new_leaf.p = p.p;
  new_leaf.max_dist = 0.;
  new_leaf.parent_dist = 0.;
  new_leaf.children = NULL;
  new_leaf.num_children = 0;
  new_leaf.scale = 100;
  return new_leaf;
}

template <class T>
double max_set(v_array< ds_node<T> > &v)
{
  double max = 0.;
  for (int i = 0; i < v.index; i++)
    if ( max < v[i].dist.last())
      max = v[i].dist.last();
  return max;
}

template <class T>
void split(v_array< ds_node<T> >& point_set, v_array< ds_node<T> >& far_set, int max_scale)
{
  unsigned int new_index = 0;
  double fmax = dist_of_scale(max_scale);
  for (int i = 0; i < point_set.index; i++){
    if (point_set[i].dist.last() <= fmax) {
      point_set[new_index++] = point_set[i];
    }
    else
      push(far_set,point_set[i]);
  }
  point_set.index=new_index;
}

template <class T>
void dist_split(v_array<ds_node <T> >& point_set,
		v_array<ds_node <T> >& new_point_set,
		T new_point,
		int max_scale,
		typename distfn<T>::Type distance,
		const double* dist_params)
{
  unsigned int new_index = 0;
  double fmax = dist_of_scale(max_scale);
  for(int i = 0; i < point_set.index; i++)
    {
      double new_d;
      new_d = distance(new_point, point_set[i].p, fmax, dist_params);
      if (new_d <= fmax ) {
	push(point_set[i].dist, new_d);
	push(new_point_set,point_set[i]);
      }
      else
	point_set[new_index++] = point_set[i];
    }
  point_set.index = new_index;
}

/*
   max_scale is the maximum scale of the node we might create here.
   point_set contains points which are 2*max_scale or less away.
*/

template<class T>
node<T> batch_insert(const ds_node<T>& p,
		  int max_scale,
		  int top_scale,
		  v_array< ds_node<T> >& point_set,
		  v_array< ds_node<T> >& consumed_set,
		  v_array<v_array< ds_node<T> > >& stack,
		  typename distfn<T>::Type distance,
		const double* dist_params)
{
  if (point_set.index == 0)
    return new_leaf(p);
  else {
    double max_dist = max_set(point_set); //O(|point_set|)
    int next_scale = min (max_scale - 1, get_scale(max_dist));
    if (top_scale - next_scale == 100) // We have points with distance 0.
      {
	v_array< node<T> > children;
	push(children,new_leaf(p));
	while (point_set.index > 0)
	  {
	    push(children,new_leaf(point_set.last()));
	    push(consumed_set,point_set.last());
	    point_set.decr();
	  }
	node<T> n = new_node(p);
	n.scale = 100; // A magic number meant to be larger than all scales.
	n.max_dist = 0;
	alloc(children,children.index);
	n.num_children = children.index;
	n.children = children.elements;
	return n;
      }
    else
      {
	v_array< ds_node<T> > far = pop(stack);
	split(point_set,far,max_scale); //O(|point_set|)

	node<T> child = batch_insert(p, next_scale, top_scale,
				  point_set, consumed_set, stack,
				  distance, dist_params);

	if (point_set.index == 0)
	  {
	    push(stack,point_set);
	    point_set=far;
	    return child;
	  }
	else {
	  node<T> n = new_node(p);
	  v_array< node<T> > children;
	  push(children, child);
	  v_array<ds_node<T> > new_point_set = pop(stack);
	  v_array<ds_node<T> > new_consumed_set = pop(stack);
	  while (point_set.index != 0) { //O(|point_set| * num_children)
	    ds_node<T> new_point_node = point_set.last();
	    T new_point = point_set.last().p;
	    double new_dist = point_set.last().dist.last();
	    push(consumed_set, point_set.last());
	    point_set.decr();

	    dist_split(point_set, new_point_set, new_point, max_scale, distance, dist_params); //O(|point_saet|)
	    dist_split(far,new_point_set,new_point,max_scale, distance, dist_params); //O(|far|)

	    node<T> new_child =
	      batch_insert(new_point_node, next_scale, top_scale,
			   new_point_set, new_consumed_set, stack,
			   distance, dist_params);
	    new_child.parent_dist = new_dist;

	    push(children, new_child);

	    double fmax = dist_of_scale(max_scale);
	    for(int i = 0; i< new_point_set.index; i++) //O(|new_point_set|)
	      {
		new_point_set[i].dist.decr();
		if (new_point_set[i].dist.last() <= fmax)
		  push(point_set, new_point_set[i]);
		else
		  push(far, new_point_set[i]);
	      }
	    for(int i = 0; i< new_consumed_set.index; i++) //O(|new_point_set|)
	      {
		new_consumed_set[i].dist.decr();
		push(consumed_set, new_consumed_set[i]);
	      }
	    new_point_set.index = 0;
	    new_consumed_set.index = 0;
	  }
	  push(stack,new_point_set);
	  push(stack,new_consumed_set);
	  push(stack,point_set);
	  point_set=far;
	  n.scale = top_scale - max_scale;
	  n.max_dist = max_set(consumed_set);
	  alloc(children,children.index);
	  n.num_children = children.index;
	  n.children = children.elements;
	  return n;
	}
      }
  }
}

template<class T> node<T> batch_create(const std::vector<T> &points,
					typename distfn<T>::Type distance,
					const double* dist_params)
{
  v_array<ds_node<T> > point_set;
  v_array<v_array<ds_node<T> > > stack;

  ds_node<T> initial_pt;
  initial_pt.p = points[0];
  for (std::vector<point>::size_type i = 1; i < points.size(); i++) {
    ds_node<T> temp;
    push(temp.dist, distance(points[0], points[i], MAXDOUBLE, dist_params));
    temp.p = points[i];
    push(point_set,temp);
  }
  v_array< ds_node < T > > consumed_set;

  double max_dist = max_set(point_set);

  node<T> top = batch_insert(initial_pt,
			  get_scale(max_dist),
			  get_scale(max_dist),
			  point_set,
			  consumed_set,
			  stack,
			  distance, dist_params);
  for (int i = 0; i<consumed_set.index;i++)
    free(consumed_set[i].dist.elements);
  free(consumed_set.elements);
  for (int i = 0; i<stack.index;i++)
    free(stack[i].elements);
  free(stack.elements);
  free(point_set.elements);
  return top;
}


#endif
