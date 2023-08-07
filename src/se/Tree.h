#pragma once

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>
#include <queue>
#include <optional>
#include <string_view>
#include <memory>
#include <stdexcept>

#include "Datatypes.h"
#include "Matrix.h"

class Node {
public:
    data_t threshold;
    unsigned int idx;
    unsigned int left, right;
    bool left_is_leaf, right_is_leaf;

    // I want to make sure that these objects are only moved and never copied. I expect the in DecisionTree / DistanceTree to 
    // not use any copy c'tors, but for safe measures we delete the copy constructor entirely.
    Node(const Node&) = default; //delete;
    Node() = default;
    Node(Node &&) = default;

    unsigned int num_bytes() const {
        return sizeof(*this);
    }
};

class Tree {
public:

    virtual void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y) = 0;

    virtual void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, std::vector<unsigned int> const &idx) = 0;

    virtual matrix2d<data_t> predict_proba(matrix2d<data_t> const &X) = 0;

    virtual unsigned int num_bytes() const = 0;

    virtual unsigned int num_nodes() const = 0;

    virtual void load(matrix1d<internal_t> const & nodes) = 0;

    virtual matrix1d<internal_t> store() const = 0;

    virtual ~Tree() { }
};