#ifndef GASE_H
#define GASE_H

#include <vector>

#include "Datatypes.h"
#include "DecisionTree.h"
#include "ShrubEnsemble.h"
#include "EnsembleRegularizer.h"
#include "TreeRegularizer.h"

class GASE {
    
private:

    TreeEnsemble * model = nullptr;

    unsigned int n_rounds;
    unsigned int n_batches;
    unsigned int n_trees;
    unsigned int init_batch_size;
    bool bootstrap; 

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
        unsigned int n_batches = 5,
        unsigned int n_rounds = 5,
        unsigned int init_batch_size = 0,
        bool bootstrap = true
    ) : n_rounds(n_rounds), n_batches(n_batches), n_trees(n_trees), init_batch_size(init_batch_size), bootstrap(bootstrap) { 
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
        } 
    }
    
    void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            return model->fit_ga(X,Y,n_trees,bootstrap,init_batch_size,n_rounds,n_batches);
        } 
    }

    void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            return model->next_ga(X,Y,n_batches);
        }
    }

    void init(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            return model->init_trees(X,Y,n_trees,bootstrap,init_batch_size);
        }
    }

    std::vector<std::vector<internal_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        if (model != nullptr) {
            return model->predict_proba(X);
        } else {
            return std::vector<std::vector<internal_t>>();
        }
    }
    
    std::vector<internal_t> weights() const {
        if (model != nullptr) {
            return model->weights();
        } else {
            return std::vector<internal_t>();
        }
    }

    unsigned int num_trees()const {
        if (model != nullptr) {
            return model->num_trees();
        } else {
            return 0;
        }
    }

    unsigned int num_bytes()const {
        if (model != nullptr) {
            return model->num_bytes();
        } else {
            return 0;
        }
    }
    
    unsigned int num_nodes() const {
        if (model != nullptr) {
            return model->num_nodes();
        } else {
            return 0;
        }
    }

    void set_weights(std::vector<internal_t> & new_weights) {
        if (model != nullptr) {
            return model->set_weights(new_weights);
        } 
    }
    
    void set_leafs(std::vector<std::vector<internal_t>> & new_leafs) {
        if (model != nullptr) {
            return model->set_leafs(new_leafs);
        } 
    }

    void set_nodes(std::vector<std::vector<Node>> & new_nodes) {
        if (model != nullptr) {
            return model->set_nodes(new_nodes);
        }
    }
    
};

#endif