/*
 *  quadtree.h
 *  Header file for a quadtree.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 */

#include <cstdlib>
#include <vector>

#ifndef SPLITTREE_H
#define SPLITREE_H

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }
static inline double abs_d(double x) { return (x <= 0 ? -x : x); }

class Cell {

public:
	double* center;
	double* width;
	int n_dims;
	bool   containsPoint(double point[]);
	~Cell() {
		delete[] center;
		delete[] width;
	}
};


class SplitTree
{

	// Fixed constants
	static const int QT_NODE_CAPACITY = 1;

	// Properties of this node in the tree
	int QT_NO_DIMS;
	bool is_leaf;
	int size;
	int cum_size;

	// Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
	Cell boundary;

	// Indices in this quad tree node, corresponding center-of-mass, and list of all children
	double* data;
	double* center_of_mass;
	int index[QT_NODE_CAPACITY];

	int num_children;
	std::vector<SplitTree*> children;
public:
	

	SplitTree(double* inp_data, int N, int no_dims);
	SplitTree(SplitTree* inp_parent, double* inp_data, double* mean_Y, double* width_Y);
	~SplitTree();
	void construct(Cell boundary);
	bool insert(int new_index);
	void subdivide();
	void computeNonEdgeForces(int point_index, double theta, double* neg_f, double* sum_Q);
private:
	
	void init(SplitTree* inp_parent, double* inp_data, double* mean_Y, double* width_Y);
	void fill(int N);
};

#endif
