#pragma once

#include <vector>

#include "Datatypes.h"
#include "DecisionTree.h"
#include "ShrubEnsemble.h"
#include "EnsembleRegularizer.h"
#include "TreeRegularizer.h"

class OSE {
    
private:

    TreeEnsemble * model = nullptr;
    
    unsigned int burnin_steps;

public:

    OSE(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        bool normalize_weights = true,
        unsigned int burnin_steps = 0,
        unsigned int max_features = 0,
        const std::string loss = "mse",
        internal_t step_size = 1e-2,
        const std::string optimizer = "sgd", 
        const std::string tree_init_mode = "train", 
        const std::string regularizer = "none",
        internal_t l_reg = 0
    ) : burnin_steps(burnin_steps) { 

        if (tree_init_mode == "random" && optimizer == "sgd" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::SGD,OPTIMIZER::OPTIMIZER_TYPE::NONE,DT::TREE_INIT::RANDOM>(n_classes, max_depth, seed, normalize_weights, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::TYPE::NO, 0.0);
        } else if (tree_init_mode == "random" && optimizer == "adam" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,OPTIMIZER::OPTIMIZER_TYPE::NONE,DT::TREE_INIT::RANDOM>(n_classes, max_depth, seed, normalize_weights, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::TYPE::NO, 0.0);
        } else if (tree_init_mode == "train" && optimizer == "sgd" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::SGD,OPTIMIZER::OPTIMIZER_TYPE::NONE,DT::TREE_INIT::TRAIN>(n_classes, max_depth, seed, normalize_weights, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::TYPE::NO, 0.0);
        } else if (tree_init_mode == "train" && optimizer == "adam" && loss == "mse") {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,OPTIMIZER::OPTIMIZER_TYPE::NONE,DT::TREE_INIT::TRAIN>(n_classes, max_depth, seed, normalize_weights, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::TYPE::NO, 0.0);
        } else if (tree_init_mode == "random" && optimizer == "sgd" && loss == "cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::SGD,OPTIMIZER::OPTIMIZER_TYPE::NONE,DT::TREE_INIT::RANDOM>(n_classes, max_depth, seed, normalize_weights, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::TYPE::NO, 0.0);
        } else if (tree_init_mode == "random" && optimizer == "adam" && loss == "cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::ADAM,OPTIMIZER::OPTIMIZER_TYPE::NONE,DT::TREE_INIT::RANDOM>(n_classes, max_depth, seed, normalize_weights, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::TYPE::NO, 0.0);
        } else if (tree_init_mode == "train" && optimizer == "sgd" && loss == "cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::SGD,OPTIMIZER::OPTIMIZER_TYPE::NONE,DT::TREE_INIT::TRAIN>(n_classes, max_depth, seed, normalize_weights, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::TYPE::NO, 0.0);
        } else if (tree_init_mode == "train" && optimizer == "adam" && loss == "cross-entropy") {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::ADAM,OPTIMIZER::OPTIMIZER_TYPE::NONE,DT::TREE_INIT::TRAIN>(n_classes, max_depth, seed, normalize_weights, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::TYPE::NO, 0.0);
        } else {
            throw std::runtime_error("Currently only the two tree_init_mode {random, train} and the two optimizer modes {adam, sgd} and the two losses {mse, cross-entropy} are supported for OSE, but you provided a combination of " + tree_init_mode + " and " + optimizer + " and " + loss);
        }
    }

    ~OSE() {
        if (model != nullptr) {
            delete model;
        }
    }

    void next(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        if (model != nullptr) {
            model->next(X,Y,burnin_steps);
        } else {
            throw std::runtime_error("The internal object pointer in OSE was null. This should now happen!");
        }
    }
    
    void init(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size) {
        if (model != nullptr) {
            model->init_trees(X,Y,n_trees, bootstrap, batch_size);
        } else {
            throw std::runtime_error("The internal object pointer in OSE was null. This should now happen!");
        }
    }

    matrix2d<data_t> predict_proba(matrix2d<data_t> const &X) {
        if (model != nullptr) {
            return model->predict_proba(X);
        } else {
            throw std::runtime_error("The internal object pointer in OSE was null. This should now happen!");
        }
    }

    unsigned int num_nodes() const {
        if (model != nullptr) {
            return model->num_nodes();
        } else {
            throw std::runtime_error("The internal object pointer in OSE was null. This should now happen!");
        }
    }

    unsigned int num_bytes() const {
        if (model != nullptr) {
            return model->num_bytes();
        } else {
            throw std::runtime_error("The internal object pointer in OSE was null. This should now happen!");
        }
    }

    unsigned int num_trees() const {
        if (model != nullptr) {
            return model->num_trees();
        } else {
            throw std::runtime_error("The internal object pointer in OSE was null. This should now happen!");
        }
    }

    // // TODO
    // void load(std::vector<std::vector<internal_t>> & new_nodes, std::vector<std::vector<internal_t>> & new_leafs, std::vector<internal_t> & new_weights) {
    //     if (model != nullptr) {
    //         model->load(new_nodes, new_leafs, new_weights);
    //     } else {
    //         throw std::runtime_error("The internal object pointer in OSE was null. This should now happen!");
    //     }
    // }

    // // TODO
    // std::tuple<std::vector<std::vector<internal_t>>, std::vector<std::vector<internal_t>>, std::vector<internal_t>> store() const {
    //     if (model != nullptr) {
    //         return model->store();
    //     } else {
    //         throw std::runtime_error("The internal object pointer in OSE was null. This should now happen!");
    //     }
    // }
};