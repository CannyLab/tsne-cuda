#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstdio>

#include "splittree.h"


// Checks whether a point lies in a cell
bool Cell::containsPoint(double point[])
{   
    for (int i = 0; i< n_dims; ++i) {
        if (abs_d(center[i] - point[i]) > width[i]) {
            return false;
        }
    }
    return true;
}


// Default constructor for quadtree -- build tree, too!
SplitTree::SplitTree(double* inp_data, int N, int no_dims)
{   
    QT_NO_DIMS = no_dims;
    num_children = 1 << no_dims;

    // Compute mean, width, and height of current map (boundaries of SplitTree)
    double* mean_Y = new double[QT_NO_DIMS]; 
    for (int d = 0; d < QT_NO_DIMS; d++) {
        mean_Y[d] = .0;
    }

    double*  min_Y = new double[QT_NO_DIMS]; 
    for (int d = 0; d < QT_NO_DIMS; d++) {
        min_Y[d] =  DBL_MAX;  
    } 
    double*  max_Y = new double[QT_NO_DIMS]; 
    for (int d = 0; d < QT_NO_DIMS; d++) {
        max_Y[d] = -DBL_MAX;
    }

    for (int n = 0; n < N; n++) {
        for (int d = 0; d < QT_NO_DIMS; d++) {
            mean_Y[d] += inp_data[n * QT_NO_DIMS + d];
            min_Y[d] = min(min_Y[d], inp_data[n * QT_NO_DIMS + d]);
            max_Y[d] = max(max_Y[d], inp_data[n * QT_NO_DIMS + d]);
        }

    }

    double* width_Y = new double[QT_NO_DIMS]; 
    for (int d = 0; d < QT_NO_DIMS; d++) {
        mean_Y[d] /= (double) N;
        width_Y[d] = max(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;    
    }

    // Construct SplitTree
    init(NULL, inp_data, mean_Y, width_Y);
    fill(N);
    delete[] max_Y; delete[] min_Y;
}

// Constructor for SplitTree with particular size and parent (do not fill the tree)
SplitTree::SplitTree(SplitTree* inp_parent, double* inp_data, double* mean_Y, double* width_Y)
{   
    QT_NO_DIMS = inp_parent->QT_NO_DIMS;
    num_children = 1 << QT_NO_DIMS;
    
    init(inp_parent, inp_data, mean_Y, width_Y);
}


// Main initialization function
void SplitTree::init(SplitTree* inp_parent, double* inp_data, double* mean_Y, double* width_Y)
{   
    // parent = inp_parent;
    data = inp_data;
    is_leaf = true;
    size = 0;
    cum_size = 0;
    
    boundary.center = mean_Y;
    boundary.width  = width_Y;
    boundary.n_dims = QT_NO_DIMS;

    index[0] = 0;

    center_of_mass = new double[QT_NO_DIMS];
    for (int i = 0; i < QT_NO_DIMS; i++) {
        center_of_mass[i] = .0;
    }
}


// Destructor for SplitTree
SplitTree::~SplitTree()
{   
    for(unsigned int i = 0; i != children.size(); i++) {
        delete children[i];
    }
    delete[] center_of_mass;
}


// Insert a point into the SplitTree
bool SplitTree::insert(int new_index)
{   
    // Ignore objects which do not belong in this quad tree
    double* point = data + new_index * QT_NO_DIMS;
    if (!boundary.containsPoint(point)) {
        return false;
    }

    // Online update of cumulative size and center-of-mass
    cum_size++;
    double mult1 = (double) (cum_size - 1) / (double) cum_size;
    double mult2 = 1.0 / (double) cum_size;
    for (int d = 0; d < QT_NO_DIMS; d++) {
        center_of_mass[d] = center_of_mass[d] * mult1 + mult2 * point[d];
    }

    // If there is space in this quad tree and it is a leaf, add the object here
    if (is_leaf && size < QT_NODE_CAPACITY) {
        index[size] = new_index;
        size++;
        return true;
    }

    // Don't add duplicates for now (this is not very nice)
    bool any_duplicate = false;
    for (int n = 0; n < size; n++) {
        bool duplicate = true;
        for (int d = 0; d < QT_NO_DIMS; d++) {
            if (point[d] != data[index[n] * QT_NO_DIMS + d]) { duplicate = false; break; }
        }
        any_duplicate = any_duplicate | duplicate;
    }
    if (any_duplicate) {
        return true;
    }

    // Otherwise, we need to subdivide the current cell
    if (is_leaf) {
        subdivide();
    }

    // Find out where the point can be inserted
    for (int i = 0; i < num_children; ++i) {
        if (children[i]->insert(new_index)) {
            return true;
        }
    }
    
    // Otherwise, the point cannot be inserted (this should never happen)
    // printf("%s\n", "No no, this should not happen");
    return false;
}

int *get_bits(int n, int bitswanted){
  int *bits = new int[bitswanted];

  int k;
  for(k=0; k<bitswanted; k++) {
    int mask =  1 << k;
    int masked_n = n & mask;
    int thebit = masked_n >> k;
    bits[k] = thebit;
  }

  return bits;
}

// Create four children which fully divide this cell into four quads of equal area
void SplitTree::subdivide() {

    // Create children
    double* new_centers = new double[2 * QT_NO_DIMS];
    for(int i = 0; i < QT_NO_DIMS; ++i) {
        new_centers[i*2]     = boundary.center[i] - .5 * boundary.width[i];
        new_centers[i*2 + 1] = boundary.center[i] + .5 * boundary.width[i];
    }

    for (int i = 0; i < num_children; ++i) {
        int *bits = get_bits(i, QT_NO_DIMS);    

        double* mean_Y = new double[QT_NO_DIMS]; 
        double* width_Y = new double[QT_NO_DIMS]; 

        // fill the means and width
        for (int d = 0; d < QT_NO_DIMS; d++) {
            mean_Y[d] = new_centers[d*2 + bits[d]];
            width_Y[d] = .5*boundary.width[d];
        }
        
        SplitTree* qt = new SplitTree(this, data, mean_Y, width_Y);        
        children.push_back(qt);
        delete[] bits; 
    }
    delete[] new_centers;

    // Move existing points to correct children
    for (int i = 0; i < size; i++) {
        // bool flag = false;
        for (int j = 0; j < num_children; j++) {
            if (children[j]->insert(index[i])) {
                // flag = true;
                break;
            }
        }
        // if (flag == false) {
        index[i] = -1;
        // }
    }
    
    // This node is not leaf now
    // Empty it
    size = 0;
    is_leaf = false;
}


// Build SplitTree on dataset
void SplitTree::fill(int N)
{
    for (int i = 0; i < N; i++) {
        insert(i);
    }
}


// Compute non-edge forces using Barnes-Hut algorithm
void SplitTree::computeNonEdgeForces(int point_index, double theta, double* neg_f, double* sum_Q)
{
    // Make sure that we spend no time on empty nodes or self-interactions
    if (cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) {
        return;
    }
    // Compute distance between point and center-of-mass
    double D = .0;
    int ind = point_index * QT_NO_DIMS;

    for (int d = 0; d < QT_NO_DIMS; d++) {
        double t  = data[ind + d] - center_of_mass[d];
        D += t * t;
    }

    // Check whether we can use this node as a "summary"
    double m = -1;
    for (int i = 0; i < QT_NO_DIMS; ++i) {
        m = max(m, boundary.width[i]);
    }
    if (is_leaf || m / sqrt(D) < theta) {

        // Compute and add t-SNE force between point and current node
        double Q = 1.0 / (1.0 + D);
        *sum_Q += cum_size * Q;
        double mult = cum_size * Q * Q;
        for (int d = 0; d < QT_NO_DIMS; d++) {
            neg_f[d] += mult * (data[ind + d] - center_of_mass[d]);
        }
    }
    else {
        // Recursively apply Barnes-Hut to children
        for (int i = 0; i < num_children; ++i) {
            children[i]->computeNonEdgeForces(point_index, theta, neg_f, sum_Q);
        }
    }
}
