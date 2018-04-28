#include "bh_tsne_ref.h"
#include "common.h"

double * BHTSNERef::computeEdgeForces(float * Xs, float * Ys, float NDIMS, float PROJDIMS, int N, int K, float sigma) {
	double * dXs = (double *) malloc(N * NDIMS * sizeof(double));
	double * dYs = (double *) malloc(N * PROJDIMS * sizeof(double));
	double * edgeForces = (double *) calloc(N * NDIMS, sizeof(double));
	for (int i = 0; i < N * NDIMS; i++) {
		dXs[i] = (double) Xs[i];
		dYs[i] = (double) Ys[i];
	}
	unsigned int * row_P;
    unsigned int * col_P;
	double * val_P;
	computeGaussianPerplexity(dXs, N, NDIMS, &row_P, &col_P, &val_P, (double) sigma, K);

	symmetrizeMatrix(&row_P, &col_P, &val_P, N);
    for(int i = 0; i < row_P[N]; i++) val_P[i] *= 12.0;

	BHTSNERef::SPTree * tree = new BHTSNERef::SPTree(PROJDIMS, dYs, N);
	tree->computeEdgeForces(row_P, col_P, val_P, N, edgeForces);
	return edgeForces;
}

double BHTSNERef::euclidean_distance(const BHTSNERef::DataPoint &t1, const BHTSNERef::DataPoint &t2){
        double dd = .0;
        double* x1 = t1._x;
        double* x2 = t2._x;
        double diff;
        for(int d = 0; d < t1._D; d++) {
            diff = (x1[d] - x2[d]);
            dd += diff * diff;
        }
        return sqrt(dd);
    };


void BHTSNERef::symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N) {

    // Get sparse matrix
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if(row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) present = true;
            }
            if(present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for(int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    unsigned int* sym_row_P = (unsigned int*) malloc((N + 1) * sizeof(unsigned int));
    unsigned int* sym_col_P = (unsigned int*) malloc(no_elem * sizeof(unsigned int));
    double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
    if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int) row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if(offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) {
                    present = true;
                    if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if(!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if(!present || (present && n <= col_P[i])) {
                offset[n]++;
                if(col_P[i] != n) offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

    double sum_P = .0;
    for(int i = 0; i < row_P[N]; i++) sum_P += sym_val_P[i];
    for(int i = 0; i < row_P[N]; i++) sym_val_P[i] /= sum_P;

    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}


void BHTSNERef::computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double sigma, int K) {

    // Allocate the memory we need
    *_col_P = (unsigned int*) calloc(N * K, sizeof(unsigned int));
    *_row_P = (unsigned int*) calloc((N + 1), sizeof(unsigned int));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;
    double* cur_P = (double*) malloc((N - 1) * sizeof(double));
    if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;

    // Build ball tree on data set
    BHTSNERef::VpTree<BHTSNERef::DataPoint, BHTSNERef::euclidean_distance>* tree = new BHTSNERef::VpTree<BHTSNERef::DataPoint, BHTSNERef::euclidean_distance>();
    std::vector<BHTSNERef::DataPoint> obj_X(N, BHTSNERef::DataPoint(D, -1, X));
    for(int n = 0; n < N; n++) obj_X[n] = BHTSNERef::DataPoint(D, n, X + n * D);
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    printf("Building tree...\n");
    std::vector<BHTSNERef::DataPoint> indices;
    std::vector<double> distances;
    for(int n = 0; n < N; n++) {

        if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);

        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);

        double beta = 1.0f / 2.0f * sigma * sigma;

        // Iterate until we found a good perplexity
        double sum_P;
       
        // Compute Gaussian kernel row
        for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

        // Compute entropy of current row
        sum_P = DBL_MIN;
        for(int m = 0; m < K; m++) sum_P += cur_P[m];

        // Row-normalize current row of P and store in matrix
        for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(unsigned int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
    }

    // Clean up memory
    obj_X.clear();
    free(cur_P);
    delete tree;
}

// Ref SPTree Implementation from Van der maaten

// Constructs cell
BHTSNERef::Cell::Cell(unsigned int inp_dimension) {
    dimension = inp_dimension;
    corner = (double*) malloc(dimension * sizeof(double));
    width  = (double*) malloc(dimension * sizeof(double));
}

BHTSNERef::Cell::Cell(unsigned int inp_dimension, double* inp_corner, double* inp_width) {
    dimension = inp_dimension;
    corner = (double*) malloc(dimension * sizeof(double));
    width  = (double*) malloc(dimension * sizeof(double));
    for(int d = 0; d < dimension; d++) setCorner(d, inp_corner[d]);
    for(int d = 0; d < dimension; d++) setWidth( d,  inp_width[d]);
}

// Destructs cell
BHTSNERef::Cell::~Cell() {
    free(corner);
    free(width);
}

double BHTSNERef::Cell::getCorner(unsigned int d) {
    return corner[d];
}

double BHTSNERef::Cell::getWidth(unsigned int d) {
    return width[d];
}

void BHTSNERef::Cell::setCorner(unsigned int d, double val) {
    corner[d] = val;
}

void BHTSNERef::Cell::setWidth(unsigned int d, double val) {
    width[d] = val;
}

// Checks whether a point lies in a cell
bool BHTSNERef::Cell::containsPoint(double point[])
{
    for(int d = 0; d < dimension; d++) {
        if(corner[d] - width[d] > point[d]) return false;
        if(corner[d] + width[d] < point[d]) return false;
    }
    return true;
}


// Default constructor for SPTree -- build tree, too!
BHTSNERef::SPTree::SPTree(unsigned int D, double* inp_data, unsigned int N)
{
    
    // Compute mean, width, and height of current map (boundaries of SPTree)
    int nD = 0;
    double* mean_Y = (double*) calloc(D,  sizeof(double));
    double*  min_Y = (double*) malloc(D * sizeof(double)); for(unsigned int d = 0; d < D; d++)  min_Y[d] =  DBL_MAX;
    double*  max_Y = (double*) malloc(D * sizeof(double)); for(unsigned int d = 0; d < D; d++)  max_Y[d] = -DBL_MAX;
    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int d = 0; d < D; d++) {
            mean_Y[d] += inp_data[n * D + d];
            if(inp_data[nD + d] < min_Y[d]) min_Y[d] = inp_data[nD + d];
            if(inp_data[nD + d] > max_Y[d]) max_Y[d] = inp_data[nD + d];
        }
        nD += D;
    }
    for(int d = 0; d < D; d++) mean_Y[d] /= (double) N;
    
    // Construct SPTree
    double* width = (double*) malloc(D * sizeof(double));
    for(int d = 0; d < D; d++) width[d] = fmax(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
    init(NULL, D, inp_data, mean_Y, width);
    fill(N);
    
    // Clean up memory
    free(mean_Y);
    free(max_Y);
    free(min_Y);
    free(width);
}


// Constructor for SPTree with particular size and parent -- build the tree, too!
BHTSNERef::SPTree::SPTree(unsigned int D, double* inp_data, unsigned int N, double* inp_corner, double* inp_width)
{
    init(NULL, D, inp_data, inp_corner, inp_width);
    fill(N);
}


// Constructor for SPTree with particular size (do not fill the tree)
BHTSNERef::SPTree::SPTree(unsigned int D, double* inp_data, double* inp_corner, double* inp_width)
{
    init(NULL, D, inp_data, inp_corner, inp_width);
}


// Constructor for SPTree with particular size and parent (do not fill tree)
BHTSNERef::SPTree::SPTree(SPTree* inp_parent, unsigned int D, double* inp_data, double* inp_corner, double* inp_width) {
    init(inp_parent, D, inp_data, inp_corner, inp_width);
}


// Constructor for SPTree with particular size and parent -- build the tree, too!
BHTSNERef::SPTree::SPTree(SPTree* inp_parent, unsigned int D, double* inp_data, unsigned int N, double* inp_corner, double* inp_width)
{
    init(inp_parent, D, inp_data, inp_corner, inp_width);
    fill(N);
}


// Main initialization function
void BHTSNERef::SPTree::init(SPTree* inp_parent, unsigned int D, double* inp_data, double* inp_corner, double* inp_width)
{
    parent = inp_parent;
    dimension = D;
    no_children = 2;
    for(unsigned int d = 1; d < D; d++) no_children *= 2;
    data = inp_data;
    is_leaf = true;
    size = 0;
    cum_size = 0;
    
    boundary = new Cell(dimension);
    for(unsigned int d = 0; d < D; d++) boundary->setCorner(d, inp_corner[d]);
    for(unsigned int d = 0; d < D; d++) boundary->setWidth( d, inp_width[d]);
    
    children = (SPTree**) malloc(no_children * sizeof(SPTree*));
    for(unsigned int i = 0; i < no_children; i++) children[i] = NULL;

    center_of_mass = (double*) malloc(D * sizeof(double));
    for(unsigned int d = 0; d < D; d++) center_of_mass[d] = .0;
    
    buff = (double*) malloc(D * sizeof(double));
}


// Destructor for SPTree
BHTSNERef::SPTree::~SPTree()
{
    for(unsigned int i = 0; i < no_children; i++) {
        if(children[i] != NULL) delete children[i];
    }
    free(children);
    free(center_of_mass);
    free(buff);
    delete boundary;
}


// Update the data underlying this tree
void BHTSNERef::SPTree::setData(double* inp_data)
{
    data = inp_data;
}


// Get the parent of the current tree
BHTSNERef::SPTree* BHTSNERef::SPTree::getParent()
{
    return parent;
}


// Insert a point into the SPTree
bool BHTSNERef::SPTree::insert(unsigned int new_index)
{
    // Ignore objects which do not belong in this quad tree
    double* point = data + new_index * dimension;
    if(!boundary->containsPoint(point))
        return false;
    
    // Online update of cumulative size and center-of-mass
    cum_size++;
    double mult1 = (double) (cum_size - 1) / (double) cum_size;
    double mult2 = 1.0 / (double) cum_size;
    for(unsigned int d = 0; d < dimension; d++) center_of_mass[d] *= mult1;
    for(unsigned int d = 0; d < dimension; d++) center_of_mass[d] += mult2 * point[d];
    
    // If there is space in this quad tree and it is a leaf, add the object here
    if(is_leaf && size < QT_NODE_CAPACITY) {
        index[size] = new_index;
        size++;
        return true;
    }
    
    // Don't add duplicates for now (this is not very nice)
    bool any_duplicate = false;
    for(unsigned int n = 0; n < size; n++) {
        bool duplicate = true;
        for(unsigned int d = 0; d < dimension; d++) {
            if(point[d] != data[index[n] * dimension + d]) { duplicate = false; break; }
        }
        any_duplicate = any_duplicate | duplicate;
    }
    if(any_duplicate) return true;
    
    // Otherwise, we need to subdivide the current cell
    if(is_leaf) subdivide();
    
    // Find out where the point can be inserted
    for(unsigned int i = 0; i < no_children; i++) {
        if(children[i]->insert(new_index)) return true;
    }
    
    // Otherwise, the point cannot be inserted (this should never happen)
    return false;
}

    
// Create four children which fully divide this cell into four quads of equal area
void BHTSNERef::SPTree::subdivide() {
    
    // Create new children
    double* new_corner = (double*) malloc(dimension * sizeof(double));
    double* new_width  = (double*) malloc(dimension * sizeof(double));
    for(unsigned int i = 0; i < no_children; i++) {
        unsigned int div = 1;
        for(unsigned int d = 0; d < dimension; d++) {
            new_width[d] = .5 * boundary->getWidth(d);
            if((i / div) % 2 == 1) new_corner[d] = boundary->getCorner(d) - .5 * boundary->getWidth(d);
            else                   new_corner[d] = boundary->getCorner(d) + .5 * boundary->getWidth(d);
            div *= 2;
        }
        children[i] = new SPTree(this, dimension, data, new_corner, new_width);
    }
    free(new_corner);
    free(new_width);
    
    // Move existing points to correct children
    for(unsigned int i = 0; i < size; i++) {
        bool success = false;
        for(unsigned int j = 0; j < no_children; j++) {
            if(!success) success = children[j]->insert(index[i]);
        }
        index[i] = -1;
    }
    
    // Empty parent node
    size = 0;
    is_leaf = false;
}


// Build SPTree on dataset
void BHTSNERef::SPTree::fill(unsigned int N)
{
    for(unsigned int i = 0; i < N; i++) insert(i);
}


// Checks whether the specified tree is correct
bool BHTSNERef::SPTree::isCorrect()
{
    for(unsigned int n = 0; n < size; n++) {
        double* point = data + index[n] * dimension;
        if(!boundary->containsPoint(point)) return false;
    }
    if(!is_leaf) {
        bool correct = true;
        for(int i = 0; i < no_children; i++) correct = correct && children[i]->isCorrect();
        return correct;
    }
    else return true;
}



// Build a list of all indices in SPTree
void BHTSNERef::SPTree::getAllIndices(unsigned int* indices)
{
    getAllIndices(indices, 0);
}


// Build a list of all indices in SPTree
unsigned int BHTSNERef::SPTree::getAllIndices(unsigned int* indices, unsigned int loc)
{
    
    // Gather indices in current quadrant
    for(unsigned int i = 0; i < size; i++) indices[loc + i] = index[i];
    loc += size;
    
    // Gather indices in children
    if(!is_leaf) {
        for(int i = 0; i < no_children; i++) loc = children[i]->getAllIndices(indices, loc);
    }
    return loc;
}


unsigned int BHTSNERef::SPTree::getDepth() {
    if(is_leaf) return 1;
    int depth = 0;
    for(unsigned int i = 0; i < no_children; i++) depth = max(depth, children[i]->getDepth());
    return 1 + depth;
}


// Compute non-edge forces using Barnes-Hut algorithm
void BHTSNERef::SPTree::computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[], double* sum_Q)
{
    
    // Make sure that we spend no time on empty nodes or self-interactions
    if(cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return;
    
    // Compute distance between point and center-of-mass
    double D = .0;
    unsigned int ind = point_index * dimension;
    for(unsigned int d = 0; d < dimension; d++) buff[d] = data[ind + d] - center_of_mass[d];
    for(unsigned int d = 0; d < dimension; d++) D += buff[d] * buff[d];
    
    // Check whether we can use this node as a "summary"
    double max_width = 0.0;
    double cur_width;
    for(unsigned int d = 0; d < dimension; d++) {
        cur_width = boundary->getWidth(d);
        max_width = (max_width > cur_width) ? max_width : cur_width;
    }
    if(is_leaf || max_width / sqrt(D) < theta) {
    
        // Compute and add t-SNE force between point and current node
        D = 1.0 / (1.0 + D);
        double mult = cum_size * D;
        *sum_Q += mult;
        mult *= D;
        for(unsigned int d = 0; d < dimension; d++) neg_f[d] += mult * buff[d];
    }
    else {

        // Recursively apply Barnes-Hut to children
        for(unsigned int i = 0; i < no_children; i++) children[i]->computeNonEdgeForces(point_index, theta, neg_f, sum_Q);
    }
}


// Computes edge forces
void BHTSNERef::SPTree::computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f)
{
    
    // Loop over all edges in the graph
    unsigned int ind1 = 0;
    unsigned int ind2 = 0;
    double D;
    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
        
            // Compute pairwise distance and Q-value
            D = 1.0;
            ind2 = col_P[i] * dimension;
            for(unsigned int d = 0; d < dimension; d++) buff[d] = data[ind1 + d] - data[ind2 + d];
            for(unsigned int d = 0; d < dimension; d++) D += buff[d] * buff[d];
            D = val_P[i] / D;
            
            // Sum positive force
            for(unsigned int d = 0; d < dimension; d++) pos_f[ind1 + d] += D * buff[d];
        }
        ind1 += dimension;
    }
}


// Print out tree
void BHTSNERef::SPTree::print() 
{
    if(cum_size == 0) {
        printf("Empty node\n");
        return;
    }

    if(is_leaf) {
        printf("Leaf node; data = [");
        for(int i = 0; i < size; i++) {
            double* point = data + index[i] * dimension;
            for(int d = 0; d < dimension; d++) printf("%f, ", point[d]);
            printf(" (index = %d)", index[i]);
            if(i < size - 1) printf("\n");
            else printf("]\n");
        }        
    }
    else {
        printf("Intersection node with center-of-mass = [");
        for(int d = 0; d < dimension; d++) printf("%f, ", center_of_mass[d]);
        printf("]; children are:\n");
        for(int i = 0; i < no_children; i++) children[i]->print();
    }
}

