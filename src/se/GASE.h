#ifndef GASE_H
#define GASE_H

#include <vector>

#include "Datatypes.h"
#include "DecisionTree.h"
#include "ShrubEnsemble.h"
#include "EnsembleRegularizer.h"
#include "TreeRegularizer.h"

class GASE {
    
protected:

    TreeEnsemble * model = nullptr;

    unsigned int const n_rounds;
    unsigned int const n_worker;
    unsigned int const n_trees;
    unsigned int const init_batch_size;
    bool const bootstrap; 

public:

    GASE(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        unsigned int max_features = 0,
        const std::string loss = "mse",
        internal_t step_size = 1e-2,
        const std::string optimizer = "mse",
        const std::string tree_init_mode = "train",
        unsigned int n_trees = 32, 
        unsigned int n_worker = 5,
        unsigned int n_rounds = 5,
        unsigned int init_batch_size = 0,
        bool bootstrap = true
    ) : n_rounds(n_rounds), n_worker(n_worker), n_trees(n_trees), init_batch_size(init_batch_size), bootstrap(bootstrap) { 
        if (tree_init_mode == "random" && optimizer == "sgd" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,DT::TREE_INIT::RANDOM>(n_classes, max_depth, seed, false,max_features, step_size,ENSEMBLE_REGULARIZER::TYPE::NO,0,TREE_REGULARIZER::TYPE::NO, 0);
        } else if (tree_init_mode == "random" && optimizer == "adam" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,DT::TREE_INIT::RANDOM>(n_classes, max_depth, seed, false,max_features, step_size,ENSEMBLE_REGULARIZER::TYPE::NO,0,TREE_REGULARIZER::TYPE::NO, 0);
        } else if (tree_init_mode == "train" && optimizer == "sgd" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,DT::TREE_INIT::TRAIN>(n_classes, max_depth, seed, false, max_features, step_size,ENSEMBLE_REGULARIZER::TYPE::NO,0,TREE_REGULARIZER::TYPE::NO, 0);
        } else if (tree_init_mode == "train" && optimizer == "adam" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,DT::TREE_INIT::TRAIN>(n_classes, max_depth, seed, false, max_features, step_size,ENSEMBLE_REGULARIZER::TYPE::NO,0,TREE_REGULARIZER::TYPE::NO, 0);
        } else if (tree_init_mode == "random" && optimizer == "sgd" && loss == "cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,DT::TREE_INIT::RANDOM>(n_classes, max_depth, seed, false, max_features, step_size,ENSEMBLE_REGULARIZER::TYPE::NO,0,TREE_REGULARIZER::TYPE::NO, 0);
        } else if (tree_init_mode == "random" && optimizer == "adam" && loss == "cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,DT::TREE_INIT::RANDOM>(n_classes, max_depth, seed, false, max_features, step_size,ENSEMBLE_REGULARIZER::TYPE::NO,0,TREE_REGULARIZER::TYPE::NO, 0);
        } else if (tree_init_mode == "train" && optimizer == "sgd" && loss == "cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,DT::TREE_INIT::TRAIN>(n_classes, max_depth, seed, false, max_features, step_size,ENSEMBLE_REGULARIZER::TYPE::NO,0,TREE_REGULARIZER::TYPE::NO, 0);
        } else if (tree_init_mode == "train" && optimizer == "adam" && loss =="cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,DT::TREE_INIT::TRAIN>(n_classes, max_depth, seed, false, max_features, step_size,ENSEMBLE_REGULARIZER::TYPE::NO,0,TREE_REGULARIZER::TYPE::NO, 0);
        } else {
            throw std::runtime_error("Currently only the two tree_init_mode {random, train}, the two optimizer modes {adam, sgd} and the two losses {mse, cross-entropy} are supported for GASE, but you provided a combination of " + tree_init_mode + " and " + optimizer + " and " + loss);
        }
    }
    
    void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            return model->fit_ga(X,Y,n_trees,bootstrap,init_batch_size,n_rounds,n_worker);
        } else {
            throw std::runtime_error("The internal object pointer in GASE was null. This should not happen!");
        }
    }

    void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            return model->next_ga(X,Y,n_worker);
        } else {
            throw std::runtime_error("The internal object pointer in GASE was null. This should not happen!");
        }
    }

    void init(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            return model->init_trees(X,Y,n_trees,bootstrap,init_batch_size);
        } else {
            throw std::runtime_error("The internal object pointer in GASE was null. This should not happen!");
        }
    }

    std::vector<std::vector<internal_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        if (model != nullptr) {
            return model->predict_proba(X);
        } else {
            throw std::runtime_error("The internal object pointer in GASE was null. This should not happen!");
        }
    }
    
    unsigned int num_nodes() const {
        if (model != nullptr) {
            return model->num_nodes();
        } else {
            throw std::runtime_error("The internal object pointer in GASE was null. This should not happen!");
        }
    }

    unsigned int num_bytes() const {
        if (model != nullptr) {
            return model->num_bytes();
        } else {
            throw std::runtime_error("The internal object pointer in GASE was null. This should not happen!");
        }
    }

    unsigned int num_trees() const {
        if (model != nullptr) {
            return model->num_trees();
        } else {
            throw std::runtime_error("The internal object pointer in GASE was null. This should now happen!");
        }
    }

    void load(std::vector<std::vector<internal_t>> & new_nodes, std::vector<std::vector<internal_t>> & new_leafs, std::vector<internal_t> & new_weights) {
        if (model != nullptr) {
            model->load(new_nodes, new_leafs, new_weights);
        } else {
            throw std::runtime_error("The internal object pointer in GASE was null. This should now happen!");
        }
    }

    std::tuple<std::vector<std::vector<internal_t>>, std::vector<std::vector<internal_t>>, std::vector<internal_t>> store() const {
        if (model != nullptr) {
            return model->store();
        } else {
            throw std::runtime_error("The internal object pointer in GASE was null. This should not happen!");
        }
    }
    
};

#endif