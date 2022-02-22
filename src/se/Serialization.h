#ifndef SE_SERIALIZATION_H
#define SE_SERIALIZATION_H

#include "ShrubEnsemble.h"

template<typename T> 
void serialize(T val, std::vector<unsigned char> &out) {
    unsigned char const * p = reinterpret_cast<unsigned char const *>(&val);

    for (unsigned int i = 0; i < sizeof(T); ++i) {
        out.push_back(p[i]);
    }    
}

template<typename T> 
T deserialize(unsigned char const * const in, unsigned int * pos) {
    union Deserializer
    {
        T const val;
        unsigned char const * const c; 

        Deserializer(unsigned char const * const c) : c(c) {};
    } tmp(in + (*pos));
    
    // Deserializer tmp;
    // tmp.c = in[i];

    *pos += sizeof(T);
    return tmp.val;

    // unsigned int i = *pos;
    // return reinterpret_cast<T>(in[i]);
}


// template <OPTIMIZER::OPTIMIZER_TYPE tree_opt, TREE_INIT tree_init>
// void serialize(Tree<tree_init, tree_opt>& tree, std::vector<unsigned char> &out) {
//     serialize(tree_opt, out);
//     serialize(tree_init, out);
//     serialize(tree.nodes.size(), out);

//     for (auto const & n : tree.nodes) {
//         serialize(n.threshold, out);
//         serialize(n.feature, out);
//         serialize(n.left, out);
//         serialize(n.right, out);
//         serialize(n.left_is_leaf, out);
//         serialize(n.right_is_leaf, out);
//     }

//     serialize(tree.leafs.size(), out);

//     for (auto const & l : tree.leafs) {
//         serialize(l, out);
//     }

//     serialize(tree.n_classes, out);
//     serialize(tree.max_depth, out);
//     serialize(tree.max_features, out);
//     serialize(tree.seed, out);
//     serialize(tree.optimizer.step_size, out);
//     // TODO do not serialize current optimizer status 
// }

// template <OPTIMIZER::OPTIMIZER_TYPE tree_opt, TREE_INIT tree_init>
// Tree<tree_init, tree_opt> deserialize(unsigned char const * const in, unsigned int * i) {
//     // OPTIMIZER::OPTIMIZER_TYPE ot = deserialize<OPTIMIZER::OPTIMIZER_TYPE>(in, i);
//     // TREE_INIT it = deserialize<TREE_INIT>(in, i);
//     size_t n_nodes = deserialize<size_t>(in, i);

//     std::vector<Node> nodes(n_nodes);
//     for (unsigned int j = 0; j < n_nodes; ++j) {
//         nodes[j].threshold = deserialize<internal_t>(in, i);
//         nodes[j].feature = deserialize<unsigned int>(in, i);
//         nodes[j].left = deserialize<unsigned int>(in, i);
//         nodes[j].right = deserialize<unsigned int>(in, i);
//         nodes[j].left_is_leaf = deserialize<bool>(in, i);
//         nodes[j].right_is_leaf = deserialize<bool>(in, i);
//     }

//     size_t n_leafs = deserialize<size_t>(in, i);
//     std::vector<internal_t> leafs(n_leafs);
//     for (unsigned int j = 0; j < n_leafs; ++j) {
//         leafs[j] = deserialize<internal_t>(in, i);
//     }
//     unsigned int n_classes = deserialize<unsigned int>(in, i);
//     unsigned int max_depth = deserialize<unsigned int>(in, i);
//     unsigned int max_features = deserialize<unsigned int>(in, i);
//     unsigned long seed = deserialize<unsigned long>(in, i);
//     internal_t step_size = deserialize<internal_t>(in, i);
    
//     Tree<tree_init, tree_opt> tree(n_classes, max_depth, max_features, seed, step_size);
//     tree.nodes = nodes;
//     tree.leafs = leafs;

//     return tree;
// }

// template <LOSS::TYPE loss_type, OPTIMIZER::OPTIMIZER_TYPE opt, OPTIMIZER::OPTIMIZER_TYPE tree_opt, TREE_INIT tree_init>
// std::vector<unsigned char> serialize(ShrubEnsemble<loss_type, opt, tree_opt, tree_init>& o) {

//     std::vector<unsigned char> out;

//     serialize(loss_type, out);
//     serialize(opt, out);
//     serialize(tree_opt, out);
//     serialize(tree_init, out);

//     serialize(o.trees.size(), out);
//     for (auto const & t: o._trees) {
//         serialize(t, out);
//     }

//     serialize(o._weights.size(), out);
//     for (auto const & w: o._weights) {
//         serialize(w, out);
//     }

//     serialize(o.n_classes, out);
//     serialize(o.max_depth, out);
//     serialize(o.seed, out);
//     serialize(o.normalize_weights, out);
//     serialize(o.burnin_steps, out);
//     serialize(o.max_features, out);
//     serialize(o.step_size, out);
//     serialize(ENSEMBLE_REGULARIZER::TYPE::NO, out);
//     serialize(o.l_ensemble_reg, out);
//     serialize(TREE_REGULARIZER::TYPE::NO, out);
//     serialize(o.l_tree_reg, out);
//     // TODO do not serialize current optimizer status
//     // TODO Add ensemble / tree_regularizer

//     return out;
// }


// GAShrubEnsembleInterface* deserialize(std::vector<unsigned char> &to_deserialize) {
//     unsigned int pos = 0; 
//     unsigned int * i = &pos;
//     auto in = &to_deserialize[0];

//     unsigned int n_classes = deserialize<unsigned int>(in, i);
//     unsigned int max_depth = deserialize<unsigned int>(in, i);
//     unsigned long seed = deserialize<unsigned long>(in, i);
//     unsigned int max_features = deserialize<unsigned int>(in, i);

//     LOSS::TYPE loss = deserialize<LOSS::TYPE>(in, i);
//     std::string s_loss;
//     if (loss == LOSS::TYPE::MSE) {
//         s_loss = "mse";
//     } else {
//         s_loss = "cross-entropy";
//     }
//     internal_t step_size = deserialize<internal_t>(in, i);

//     OPTIMIZER::OPTIMIZER_TYPE optimizer = deserialize<OPTIMIZER::OPTIMIZER_TYPE>(in, i);
//     std::string s_opt;
//     if (optimizer == OPTIMIZER::OPTIMIZER_TYPE::NONE) {
//         s_opt = "none";
//     } else if (optimizer == OPTIMIZER::OPTIMIZER_TYPE::SGD) {
//         s_opt = "sgd";
//     } else {
//         s_opt = "adam";
//     }

//     TREE_INIT tree_init_mode = deserialize<TREE_INIT>(in, i);
//     std::string s_tree_init_mode;
//     if (tree_init_mode == TREE_INIT::RANDOM) {
//         s_tree_init_mode = "random";
//     } else {
//         s_tree_init_mode = "train";
//     }

//     unsigned int n_trees = deserialize<unsigned int>(in, i);
//     unsigned int n_batches = deserialize<unsigned int>(in, i);
//     unsigned int n_rounds = deserialize<unsigned int>(in, i);
//     unsigned int init_batch_size = deserialize<unsigned int>(in, i);
//     bool bootstrap = deserialize<bool>(in, i);

//     GAShrubEnsembleAdaptor tmp(
//         n_classes, 
//         max_depth,
//         seed,
//         max_features,
//         s_loss,
//         step_size,
//         s_optimizer,
//         s_tree_init_mode,
//         n_trees, 
//         n_batches,
//         n_rounds,
//         init_batch_size,
//         bootstrap
//     ); 


//     return tmp->model;

//     // bool normalize_weights = deserialize<bool>(in, i);
//     // unsigned int burnin_steps = deserialize<unsigned int>(in, i);
//     // unsigned int max_features = deserialize<unsigned int>(in, i);


//     // TREE_INIT tree_init_mode = deserialize<TREE_INIT>(in, i);

//     // OPTIMIZER::OPTIMIZER_TYPE tot = deserialize<OPTIMIZER::OPTIMIZER_TYPE>(in, i);

//     // size_t n_trees = deserialize<size_t>(in, i);

//     // // if (it == TREE_INIT::RANDOM && tot == OPTIMIZER::OPTIMIZER_TYPE::NONE) {

//     // // }

//     // std::vector<Tree<it,tot>> trees(n_trees);
//     // for (unsigned int j = 0; j < n_trees; ++j) {
//     //     trees[j] = deserialize<Tree<it,tot>>(in, i);
//     // }

//     // size_t n_weights = deserialize<size_t>(in, i);
//     // std::vector<internal_t> weights(n_weights);
//     // for (unsigned int j = 0; j < n_weights; ++j) {
//     //     weights[j] = deserialize<internal_t>(in, i);
//     // }

    

//     // internal_t step_size = deserialize<internal_t>(in, i);
//     // ENSEMBLE_REGULARIZER::TYPE e_reg = deserialize<TREE_REGULARIZER::TYPE>(in, i);
//     // internal_t l_ensemble_reg = deserialize<internal_t>(in, i);
//     // TREE_REGULARIZER::TYPE t_reg = deserialize<TREE_REGULARIZER::TYPE>(in, i);
//     // internal_t l_tree_reg = deserialize<internal_t>(in, i);

//     // auto ga = new GAShrubEnsemble<lt, ot, it>(n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, step_size, e_reg, l_ensemble_reg, t_reg, l_tree_reg);
//     // ga->_tree = trees;
//     // ga->_weights = weights;

//     // return ga;
// }
#endif