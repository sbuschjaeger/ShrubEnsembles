#pragma once

#include <vector>

#include "Datatypes.h"
#include "Matrix.h"
#include "DecisionTree.h"
#include "ShrubEnsemble.h"
#include "EnsembleRegularizer.h"
#include "TreeRegularizer.h"

class MASE {
    
protected:

    TreeEnsemble * model = nullptr;

    unsigned int const n_trees;
    unsigned int const n_rounds;
    unsigned int const batch_size;
    bool const bootstrap; 
    unsigned int const burnin_steps;
    unsigned int const n_worker;
    std::string const optimizer;
    std::string const tree_init_mode;
    unsigned int const init_tree_size;

public:

    MASE(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        unsigned int burnin_steps = 0,
        unsigned int max_features = 0,
        const std::string loss = "mse",
        internal_t step_size = 1e-2,
        const std::string optimizer = "sgd",
        const std::string tree_init_mode = "train",
        unsigned int n_trees = 32, 
        unsigned int n_worker = 8, 
        unsigned int n_rounds = 5,
        unsigned int batch_size = 0,
        bool bootstrap = true, 
        unsigned int init_tree_size = 0
    ) : n_trees(n_trees), n_rounds(n_rounds), batch_size(batch_size), bootstrap(bootstrap), burnin_steps(burnin_steps), n_worker(n_worker), optimizer(optimizer), tree_init_mode(tree_init_mode), init_tree_size(init_tree_size) { 
       
        if (tree_init_mode == "random" && optimizer == "sgd" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,DT::TREE_INIT::RANDOM>( n_classes, max_depth, seed,  false, max_features, step_size, ENSEMBLE_REGULARIZER::TYPE::NO, 0, TREE_REGULARIZER::TYPE::NO, 0 );
        } else if (tree_init_mode == "random" && optimizer == "adam" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,DT::TREE_INIT::RANDOM>( n_classes, max_depth, seed,  false, max_features, step_size, ENSEMBLE_REGULARIZER::TYPE::NO, 0, TREE_REGULARIZER::TYPE::NO, 0 );
        } else if (tree_init_mode == "train" && optimizer == "sgd" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,DT::TREE_INIT::TRAIN>( n_classes, max_depth, seed,  false, max_features, step_size, ENSEMBLE_REGULARIZER::TYPE::NO, 0, TREE_REGULARIZER::TYPE::NO, 0 );
        } else if (tree_init_mode == "train" && optimizer == "adam" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,DT::TREE_INIT::TRAIN>( n_classes, max_depth, seed,  false, max_features, step_size, ENSEMBLE_REGULARIZER::TYPE::NO, 0, TREE_REGULARIZER::TYPE::NO, 0 );
        } else if (tree_init_mode == "random" && optimizer == "sgd" && loss == "cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,DT::TREE_INIT::RANDOM>( n_classes, max_depth, seed,  false, max_features, step_size, ENSEMBLE_REGULARIZER::TYPE::NO, 0, TREE_REGULARIZER::TYPE::NO, 0 );
        } else if (tree_init_mode == "random" && optimizer == "adam" && loss == "cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,DT::TREE_INIT::RANDOM>( n_classes, max_depth, seed,  false, max_features, step_size, ENSEMBLE_REGULARIZER::TYPE::NO, 0, TREE_REGULARIZER::TYPE::NO, 0 );
        } else if (tree_init_mode == "train" && optimizer == "sgd" && loss == "cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,DT::TREE_INIT::TRAIN>( n_classes, max_depth, seed,  false, max_features, step_size, ENSEMBLE_REGULARIZER::TYPE::NO, 0, TREE_REGULARIZER::TYPE::NO, 0 );
        } else if (tree_init_mode == "train" && optimizer == "adam" && loss == "cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,DT::TREE_INIT::TRAIN>( n_classes, max_depth, seed,  false, max_features, step_size, ENSEMBLE_REGULARIZER::TYPE::NO, 0, TREE_REGULARIZER::TYPE::NO, 0 );
        } else {
            throw std::runtime_error("Currently only the two tree_init_mode {random, train}, the two optimizer modes {adam, sgd} and the two losses {mse, cross-entropy} are supported for MASE, but you provided a combination of " + tree_init_mode + " and " + optimizer + " and " + loss);
        }
    }

    ~MASE() {
        if (model != nullptr) {
            delete model;
        }
    }

    void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        if (model != nullptr) {
            model->fit_ma(X, Y, n_trees, init_tree_size, n_worker, bootstrap,batch_size, n_rounds, burnin_steps);
        } else {
            throw std::runtime_error("The internal object pointer in MASE was null. This should not happen!");
        }
    }

    void next(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        if (model != nullptr) {
            if (n_worker == 1) {
                model->update_trees(X, Y, burnin_steps,batch_size, bootstrap);
            } else {
                model->next_ma(X,Y,n_worker,bootstrap, batch_size, burnin_steps);
            }
        } else {
            throw std::runtime_error("The internal object pointer in MASE was null. This should not happen!");
        }
    }

    void init(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        if (model != nullptr) {
            if (init_tree_size == 0 || init_tree_size > X.rows) {
                model->init_trees(X,Y,n_trees,bootstrap,X.rows);
            } else {
                model->init_trees(X,Y,n_trees,bootstrap,init_tree_size);
            }
        } else {
            throw std::runtime_error("The internal object pointer in MASE was null. This should not happen!");
        }
    }

    matrix2d<data_t> predict_proba(matrix2d<data_t> const &X) {
        if (model != nullptr) {
            return model->predict_proba(X);
        } else {
            throw std::runtime_error("The internal object pointer in MASE was null. This should not happen!");
        }
    }

    unsigned int num_nodes() const {
        if (model != nullptr) {
            return model->num_nodes();
        } else {
            throw std::runtime_error("The internal object pointer in MASE was null. This should not happen!");
        }
    }

    unsigned int num_bytes() const {
        if (model != nullptr) {
            return model->num_bytes();
        } else {
            throw std::runtime_error("The internal object pointer in MASE was null. This should not happen!");
        }
    }

    unsigned int num_trees() const {
        if (model != nullptr) {
            return model->num_trees();
        } else {
            throw std::runtime_error("The internal object pointer in MASE was null. This should not happen!");
        }
    }

    void load(std::vector<matrix1d<internal_t>> & new_nodes, std::vector<matrix1d<internal_t>> & new_leafs, std::vector<internal_t> & new_weights) {
        if (model != nullptr) {
            model->load(new_nodes, new_leafs, new_weights);
        } else {
            throw std::runtime_error("The internal object pointer in MASE was null. This should not happen!");
        }
    }

    std::tuple<std::vector<matrix1d<internal_t>>, std::vector<matrix1d<internal_t>>, std::vector<internal_t>> store() const {
        if (model != nullptr) {
            return model->store();
        } else {
            throw std::runtime_error("The internal object pointer in MASE was null. This should not happen!");
        }
    }
};