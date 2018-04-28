/**
 * @brief Reference Barnes-Hut T-SNE implementation on the CPU
  * 
 * @file bh_tsne_ref.h
 * @author Forrest Huang
 * @date 2018-04-28
 */

#ifndef BH_TSNE_REF_H
#define BH_TSNE_REF_H


namespace BHTSNERef {
	double * computeEdgeForces(float * Xs, float * Ys, float PROJDIMS, int N);

	double * computeNonEdgeForces();
	class Cell {

	    unsigned int dimension;
	    double* corner;
	    double* width;
	    
	    
		public:
		    Cell(unsigned int inp_dimension);
		    Cell(unsigned int inp_dimension, double* inp_corner, double* inp_width);
		    ~Cell();
		    
		    double getCorner(unsigned int d);
		    double getWidth(unsigned int d);
		    void setCorner(unsigned int d, double val);
		    void setWidth(unsigned int d, double val);
		    bool containsPoint(double point[]);
	};


	class SPTree {
	    
	    // Fixed constants
	    static const unsigned int QT_NODE_CAPACITY = 1;

	    // A buffer we use when doing force computations
	    double* buff;
	    
	    // Properties of this node in the tree
	    SPTree* parent;
	    unsigned int dimension;
	    bool is_leaf;
	    unsigned int size;
	    unsigned int cum_size;
	        
	    // Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
	    Cell* boundary;
	    
	    // Indices in this space-partitioning tree node, corresponding center-of-mass, and list of all children
	    double* data;
	    double* center_of_mass;
	    unsigned int index[QT_NODE_CAPACITY];
	    
	    // Children
	    SPTree** children;
	    unsigned int no_children;
	    
	public:
	    SPTree(unsigned int D, double* inp_data, unsigned int N);
	    SPTree(unsigned int D, double* inp_data, double* inp_corner, double* inp_width);
	    SPTree(unsigned int D, double* inp_data, unsigned int N, double* inp_corner, double* inp_width);
	    SPTree(SPTree* inp_parent, unsigned int D, double* inp_data, unsigned int N, double* inp_corner, double* inp_width);
	    SPTree(SPTree* inp_parent, unsigned int D, double* inp_data, double* inp_corner, double* inp_width);
	    ~SPTree();
	    void setData(double* inp_data);
	    SPTree* getParent();
	    void construct(Cell boundary);
	    bool insert(unsigned int new_index);
	    void subdivide();
	    bool isCorrect();
	    void rebuildTree();
	    void getAllIndices(unsigned int* indices);
	    unsigned int getDepth();
	    void computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[], double* sum_Q);
	    void computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, int N, double* pos_f);
	    void print();
	    
	private:
	    void init(SPTree* inp_parent, unsigned int D, double* inp_data, double* inp_corner, double* inp_width);
	    void fill(unsigned int N);
	    unsigned int getAllIndices(unsigned int* indices, unsigned int loc);
	    bool isChild(unsigned int test_index, unsigned int start, unsigned int end);
	};

	
	class DataPoint
	{
	    int _ind;

		public:
		    double* _x;
		    int _D;
		    DataPoint() {
		        _D = 1;
		        _ind = -1;
		        _x = NULL;
		    }
		    DataPoint(int D, int ind, double* x) {
		        _D = D;
		        _ind = ind;
		        _x = (double*) malloc(_D * sizeof(double));
		        for(int d = 0; d < _D; d++) _x[d] = x[d];
		    }
		    DataPoint(const DataPoint& other) {                     // this makes a deep copy -- should not free anything
		        if(this != &other) {
		            _D = other.dimensionality();
		            _ind = other.index();
		            _x = (double*) malloc(_D * sizeof(double));      
		            for(int d = 0; d < _D; d++) _x[d] = other.x(d);
		        }
		    }
		    ~DataPoint() { if(_x != NULL) free(_x); }
		    DataPoint& operator= (const DataPoint& other) {         // asignment should free old object
		        if(this != &other) {
		            if(_x != NULL) free(_x);
		            _D = other.dimensionality();
		            _ind = other.index();
		            _x = (double*) malloc(_D * sizeof(double));
		            for(int d = 0; d < _D; d++) _x[d] = other.x(d);
		        }
		        return *this;
		    }
		    int index() const { return _ind; }
		    int dimensionality() const { return _D; }
		    double x(int d) const { return _x[d]; }
	};

	double euclidean_distance(const DataPoint &t1, const DataPoint &t2) {
	    double dd = .0;
	    double* x1 = t1._x;
	    double* x2 = t2._x;
	    double diff;
	    for(int d = 0; d < t1._D; d++) {
	        diff = (x1[d] - x2[d]);
	        dd += diff * diff;
	    }
	    return sqrt(dd);
	}


	template<typename T, double (*distance)( const T&, const T& )>
	class VpTree
	{
		public:
		    
		    // Default constructor
		    VpTree() : _root(0) {}
		    
		    // Destructor
		    ~VpTree() {
		        delete _root;
		    }

		    // Function to create a new VpTree from data
		    void create(const std::vector<T>& items) {
		        delete _root;
		        _items = items;
		        _root = buildFromPoints(0, items.size());
		    }
		    
		    // Function that uses the tree to find the k nearest neighbors of target
		    void search(const T& target, int k, std::vector<T>* results, std::vector<double>* distances)
		    {
		        
		        // Use a priority queue to store intermediate results on
		        std::priority_queue<HeapItem> heap;
		        
		        // Variable that tracks the distance to the farthest point in our results
		        _tau = DBL_MAX;
		        
		        // Perform the search
		        search(_root, target, k, heap);
		        
		        // Gather final results
		        results->clear(); distances->clear();
		        while(!heap.empty()) {
		            results->push_back(_items[heap.top().index]);
		            distances->push_back(heap.top().dist);
		            heap.pop();
		        }
		        
		        // Results are in reverse order
		        std::reverse(results->begin(), results->end());
		        std::reverse(distances->begin(), distances->end());
		    }
		    
		private:
		    std::vector<T> _items;
		    double _tau;
		    
		    // Single node of a VP tree (has a point and radius; left children are closer to point than the radius)
		    struct Node
		    {
		        int index;              // index of point in node
		        double threshold;       // radius(?)
		        Node* left;             // points closer by than threshold
		        Node* right;            // points farther away than threshold
		        
		        Node() :
		        index(0), threshold(0.), left(0), right(0) {}
		        
		        ~Node() {               // destructor
		            delete left;
		            delete right;
		        }
		    }* _root;
		    
		    
		    // An item on the intermediate result queue
		    struct HeapItem {
		        HeapItem( int index, double dist) :
		        index(index), dist(dist) {}
		        int index;
		        double dist;
		        bool operator<(const HeapItem& o) const {
		            return dist < o.dist;
		        }
		    };
		    
		    // Distance comparator for use in std::nth_element
		    struct DistanceComparator
		    {
		        const T& item;
		        DistanceComparator(const T& item) : item(item) {}
		        bool operator()(const T& a, const T& b) {
		            return distance(item, a) < distance(item, b);
		        }
		    };
		    
		    // Function that (recursively) fills the tree
		    Node* buildFromPoints( int lower, int upper )
		    {
		        if (upper == lower) {     // indicates that we're done here!
		            return NULL;
		        }
		        
		        // Lower index is center of current node
		        Node* node = new Node();
		        node->index = lower;
		        
		        if (upper - lower > 1) {      // if we did not arrive at leaf yet
		            
		            // Choose an arbitrary point and move it to the start
		            int i = (int) ((double)rand() / RAND_MAX * (upper - lower - 1)) + lower;
		            std::swap(_items[lower], _items[i]);
		            
		            // Partition around the median distance
		            int median = (upper + lower) / 2;
		            std::nth_element(_items.begin() + lower + 1,
		                             _items.begin() + median,
		                             _items.begin() + upper,
		                             DistanceComparator(_items[lower]));
		            
		            // Threshold of the new node will be the distance to the median
		            node->threshold = distance(_items[lower], _items[median]);
		            
		            // Recursively build tree
		            node->index = lower;
		            node->left = buildFromPoints(lower + 1, median);
		            node->right = buildFromPoints(median, upper);
		        }
		        
		        // Return result
		        return node;
		    }
		    
		    // Helper function that searches the tree    
		    void search(Node* node, const T& target, int k, std::priority_queue<HeapItem>& heap)
		    {
		        if(node == NULL) return;     // indicates that we're done here
		        
		        // Compute distance between target and current node
		        double dist = distance(_items[node->index], target);

		        // If current node within radius tau
		        if(dist < _tau) {
		            if(heap.size() == k) heap.pop();                 // remove furthest node from result list (if we already have k results)
		            heap.push(HeapItem(node->index, dist));           // add current node to result list
		            if(heap.size() == k) _tau = heap.top().dist;     // update value of tau (farthest point in result list)
		        }
		        
		        // Return if we arrived at a leaf
		        if(node->left == NULL && node->right == NULL) {
		            return;
		        }
		        
		        // If the target lies within the radius of ball
		        if(dist < node->threshold) {
		            if(dist - _tau <= node->threshold) {         // if there can still be neighbors inside the ball, recursively search left child first
		                search(node->left, target, k, heap);
		            }
		            
		            if(dist + _tau >= node->threshold) {         // if there can still be neighbors outside the ball, recursively search right child
		                search(node->right, target, k, heap);
		            }
		        
		        // If the target lies outsize the radius of the ball
		        } else {
		            if(dist + _tau >= node->threshold) {         // if there can still be neighbors outside the ball, recursively search right child first
		                search(node->right, target, k, heap);
		            }
		            
		            if (dist - _tau <= node->threshold) {         // if there can still be neighbors inside the ball, recursively search left child
		                search(node->left, target, k, heap);
		            }
		        }
		    }
	};

}