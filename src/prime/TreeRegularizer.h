#ifndef TREE_REGULARIZER_H
#define TREE_REGULARIZER_H

#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>

#include "Tree.h"
#include "Datatypes.h"

enum class TREE_REGULARIZER {NO,NODES};

template <TREE_INIT tree_init, TREE_NEXT tree_next, typename pred_t>
data_t tree_no_reg(Tree<tree_init, tree_next, pred_t> const &tree) {
    return 0.0;
}

template <TREE_INIT tree_init, TREE_NEXT tree_next, typename pred_t>
data_t tree_nodes_reg(Tree<tree_init, tree_next, pred_t> const &tree) {
    return tree.get_num_nodes();
}

// https://stackoverflow.com/questions/14848924/how-to-define-typedef-of-function-pointer-which-has-template-arguments
template <TREE_INIT tree_init, TREE_NEXT tree_next, typename pred_t>
using fptr = data_t (*)(Tree<tree_init, tree_next, pred_t> const &tree);

template <TREE_INIT tree_init, TREE_NEXT tree_next, typename pred_t>
fptr<tree_init, tree_next, pred_t> tree_reg_from_enum(TREE_REGULARIZER reg) {
    if (reg == TREE_REGULARIZER::NO) {
        return tree_no_reg;
    } else if (reg == TREE_REGULARIZER::NODES) {
        return tree_nodes_reg;
    } else {
        throw std::runtime_error("Wrong regularizer enum provided. No implementation for this enum found");
    }
}

auto tree_regularizer_from_string(std::string const & regularizer) {
    if (regularizer == "none" || regularizer == "no") {
        return TREE_REGULARIZER::NO;
    } else if (regularizer  == "NODES" || regularizer == "nodes") {
        return TREE_REGULARIZER::NODES;
    } else {
        throw std::runtime_error("Currently only the regularizer {none, nodes} are supported, but you provided: " + regularizer);
    }
}

#endif