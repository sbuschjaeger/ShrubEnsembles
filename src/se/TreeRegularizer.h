#pragma once

#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>

#include "Optimizer.h"
#include "Tree.h"
#include "Datatypes.h"

namespace TREE_REGULARIZER {

enum class TYPE {NO,NODES};

template <OPTIMIZER::OPTIMIZER_TYPE optimizer_type>
data_t no_reg(Tree<optimizer_type> const & tree) {
    return 0.0;
}

template <OPTIMIZER::OPTIMIZER_TYPE optimizer_type>
data_t nodes_reg(Tree<optimizer_type> const & tree) {
    return tree.num_nodes();
}

// https://stackoverflow.com/questions/14848924/how-to-define-typedef-of-function-pointer-which-has-template-arguments
template <OPTIMIZER::OPTIMIZER_TYPE optimizer_type>
using fptr = internal_t (*)(Tree<optimizer_type> const &);

// template <DT::TREE_INIT tree_init, OPTIMIZER::OPTIMIZER_TYPE tree_opt>
template <OPTIMIZER::OPTIMIZER_TYPE optimizer_type>
fptr<optimizer_type> from_enum(TYPE reg) {
    if (reg == TYPE::NO) {
        return no_reg;
    } else if (reg == TYPE::NODES) {
        return nodes_reg;
    } else {
        throw std::runtime_error("Wrong regularizer enum provided. No implementation for this enum found");
    }
}

auto from_string(std::string const & regularizer) {
    if (regularizer == "none" || regularizer == "no") {
        return TYPE::NO;
    } else if (regularizer  == "NODES" || regularizer == "nodes") {
        return TYPE::NODES;
    } else {
        throw std::runtime_error("Currently only the regularizer {none, nodes} are supported, but you provided: " + regularizer);
    }
}

}