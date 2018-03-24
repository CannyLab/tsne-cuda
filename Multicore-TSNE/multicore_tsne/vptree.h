/*
 *  vptree.h
 *  Implementation of a vantage-point tree.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 */


#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <queue>
#include <limits>


#ifndef VPTREE_H
#define VPTREE_H

class DataPoint
{
    int _D;
    int _ind;

public:
    double* _x;
    DataPoint() {
        _D = 1;
        _ind = -1;
        _x = NULL;
    }
    DataPoint(int D, int ind, double* x) {
        _D = D;
        _ind = ind;
        _x = x;
    }
    DataPoint(const DataPoint& other) {                     // this makes a deep copy -- should not free anything
        if (this != &other) {
            _D = other.dimensionality();
            _ind = other.index();
            _x = other._x;
        }
    }

    DataPoint& operator= (const DataPoint& other) {         // asignment should free old object
        if (this != &other) {
            _D = other.dimensionality();
            _ind = other.index();
            _x = other._x;
        }
        return *this;
    }
    int index() const { return _ind; }
    int dimensionality() const { return _D; }
    double x(int d) const { return _x[d]; }
};


double euclidean_distance_squared(const DataPoint &t1, const DataPoint &t2) {
    double dd = .0;
    for (int d = 0; d < t1.dimensionality(); d++) {
        double t = (t1.x(d) - t2.x(d));
        dd += t * t;
    }
    return dd;
}

inline double euclidean_distance(const DataPoint &t1, const DataPoint &t2) {
    return sqrt(euclidean_distance_squared(t1, t2));
}


template<typename T, double (*distance)( const DataPoint&, const DataPoint&)>
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
        double tau = DBL_MAX;

        // Perform the searcg
        search(_root, target, k, heap, tau);

        // Gather final results
        results->clear(); distances->clear();
        while (!heap.empty()) {
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
        explicit DistanceComparator(const T& item) : item(item) {}
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
    void search(Node* node, const T& target, unsigned int k, std::priority_queue<HeapItem>& heap, double& tau)
    {
        if (node == NULL) return;    // indicates that we're done here

        // Compute distance between target and current node
        double dist = distance(_items[node->index], target);

        // If current node within radius tau
        if (dist < tau) {
            if (heap.size() == k) heap.pop();                // remove furthest node from result list (if we already have k results)
            heap.push(HeapItem(node->index, dist));           // add current node to result list
            if (heap.size() == k) tau = heap.top().dist;    // update value of tau (farthest point in result list)
        }

        // Return if we arrived at a leaf
        if (node->left == NULL && node->right == NULL) {
            return;
        }

        // If the target lies within the radius of ball
        if (dist < node->threshold) {
            if (dist - tau <= node->threshold) {        // if there can still be neighbors inside the ball, recursively search left child first
                search(node->left, target, k, heap, tau);
            }

            if (dist + tau >= node->threshold) {        // if there can still be neighbors outside the ball, recursively search right child
                search(node->right, target, k, heap, tau);
            }

            // If the target lies outsize the radius of the ball
        } else {
            if (dist + tau >= node->threshold) {        // if there can still be neighbors outside the ball, recursively search right child first
                search(node->right, target, k, heap, tau);
            }

            if (dist - tau <= node->threshold) {         // if there can still be neighbors inside the ball, recursively search left child
                search(node->left, target, k, heap, tau);
            }
        }
    }
};

#endif
