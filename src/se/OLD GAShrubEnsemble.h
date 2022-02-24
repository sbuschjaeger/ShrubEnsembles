#ifndef GA_SHRUB_ENSEMBLE_H
#define GA_SHRUB_ENSEMBLE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>

#include "ShrubEnsemble.h"
#include "Optimizer.h"
#include "Serialization.h"

class GAShrubEnsemble {
    
private:

    TreeEnsemble * model = nullptr;

    unsigned int n_rounds;
    unsigned int n_batches;
    unsigned int n_trees;
    unsigned int init_batch_size;
    bool bootstrap; 

public:

    GAShrubEnsemble(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        unsigned int max_features = 0,
        LOSS::TYPE loss = LOSS::TYPE::MSE, 
        internal_t step_size = 1e-2,
        OPTIMIZER::OPTIMIZER_TYPE optimizer = OPTIMIZER::OPTIMIZER_TYPE::SGD, 
        TREE_INIT tree_init_mode = TREE_INIT::TRAIN,
        unsigned int n_trees = 32, 
        unsigned int n_batches = 5,
        unsigned int n_rounds = 5,
        unsigned int init_batch_size = 0,
        bool bootstrap = true
    ) : n_rounds(n_rounds), n_batches(n_batches), n_trees(n_trees), init_batch_size(init_batch_size), bootstrap(bootstrap) { 
        if (tree_init_mode == TREE_INIT::RANDOM && optimizer == OPTIMIZER::OPTIMIZER_TYPE::SGD && loss == LOSS::TYPE::MSE) {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::RANDOM>(n_classes, max_depth, seed, false, 0,max_features, step_size);
        } else if (tree_init_mode == TREE_INIT::RANDOM && optimizer == OPTIMIZER::OPTIMIZER_TYPE::ADAM && loss == LOSS::TYPE::MSE) {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::RANDOM>(n_classes, max_depth, seed, false, 0,max_features, step_size);
        } else if (tree_init_mode == TREE_INIT::TRAIN && optimizer == OPTIMIZER::OPTIMIZER_TYPE::SGD && loss == LOSS::TYPE::MSE) {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::TRAIN>(n_classes, max_depth, seed, false, 0,max_features, step_size);
        } else if (tree_init_mode == TREE_INIT::TRAIN && optimizer == OPTIMIZER::OPTIMIZER_TYPE::ADAM && loss == LOSS::TYPE::MSE) {
            model = new ShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::TRAIN>(n_classes, max_depth, seed, false, 0,max_features, step_size);
        } else if (tree_init_mode == TREE_INIT::RANDOM && optimizer == OPTIMIZER::OPTIMIZER_TYPE::SGD && loss == LOSS::TYPE::CROSS_ENTROPY) {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::RANDOM>(n_classes, max_depth, seed, false, 0,max_features, step_size);
        } else if (tree_init_mode == TREE_INIT::RANDOM && optimizer == OPTIMIZER::OPTIMIZER_TYPE::ADAM && loss == LOSS::TYPE::CROSS_ENTROPY) {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::RANDOM>(n_classes, max_depth, seed, false, 0,max_features, step_size);
        } else if (tree_init_mode == TREE_INIT::TRAIN && optimizer == OPTIMIZER::OPTIMIZER_TYPE::SGD && loss == LOSS::TYPE::CROSS_ENTROPY) {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::TRAIN>(n_classes, max_depth, seed, false, 0,max_features, step_size);
        } else if (tree_init_mode == TREE_INIT::TRAIN && optimizer == OPTIMIZER::OPTIMIZER_TYPE::ADAM && loss == LOSS::TYPE::CROSS_ENTROPY) {
            model = new ShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::NONE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::TRAIN>(n_classes, max_depth, seed, false, 0,max_features, step_size);
        } 
    }

    GAShrubEnsemble(
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
    ) { 
        LOSS::TYPE lt = LOSS::from_string(loss);
        OPTIMIZER::OPTIMIZER_TYPE ot = OPTIMIZER::optimizer_from_string(optimizer);
        TREE_INIT ti = tree_init_from_string(tree_init_mode);
        
        GAShrubEnsemble(n_classes, max_depth, seed, max_features, lt, step_size, ot, ti, n_trees, n_batches, n_rounds, init_batch_size, bootstrap);
    }

    GAShrubEnsemble(std::vector<unsigned char> & ga_string) {
        /*
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        unsigned int max_features = 0,
        LOSS::TYPE loss = LOSS::TYPE::MSE, 
        internal_t step_size = 1e-2,
        OPTIMIZER::OPTIMIZER_TYPE optimizer = OPTIMIZER::OPTIMIZER_TYPE::SGD, 
        TREE_INIT tree_init_mode = TREE_INIT::TRAIN,
        unsigned int n_trees = 32, 
        unsigned int n_batches = 5,
        unsigned int n_rounds = 5,
        unsigned int init_batch_size = 0,
        bool bootstrap = true
        */
        unsigned int pos = 0; 
        unsigned int * i = &pos;
        auto in = &ga_string[0];

        unsigned int n_classes = deserialize<unsigned int>(in, i);
        unsigned int max_depth = deserialize<unsigned int>(in, i);
        unsigned long seed = deserialize<unsigned long>(in, i);
        unsigned int max_features = deserialize<unsigned int>(in, i);

        LOSS::TYPE loss = deserialize<LOSS::TYPE>(in, i);
        internal_t step_size = deserialize<internal_t>(in, i);
        OPTIMIZER::OPTIMIZER_TYPE optimizer = deserialize<OPTIMIZER::OPTIMIZER_TYPE>(in, i);
        TREE_INIT tree_init_mode = deserialize<TREE_INIT::TRAIN>(in, i);

        unsigned int n_trees = deserialize<unsigned int>(in, i);
        unsigned int n_batches = deserialize<unsigned int>(in, i);
        unsigned int n_rounds = deserialize<unsigned int>(in, i);
        unsigned int init_batch_size = deserialize<unsigned int>(in, i);
        bool bootstrap = deserialize<bool>(in, i);

        GAShrubEnsemble(n_classes, max_depth, seed, max_features, loss, step_size, optimizer, tree_init_mode, n_trees, n_batches, n_rounds, init_batch_size, bootstrap);       

        std::vector<TreeInterface> trees(n_trees);
        for (unsigned int j = 0;j < n_trees; ++j) {
            trees[j] = TreeInterface(in, i);
            // TODO Implement proper tree de-serialization
        }
        
        std::vector<internal_t> weights(n_trees);
        for (unsigned int j = 0;j < n_trees; ++j) {
            weights[j] = deserialize<internal_t>(in, i);
        }
    }

    std::vector<unsigned char> to_string() const {
        std::vector<unsigned char> ga_string;

        serialize(n_rounds, ga_string);
        serialize(n_batches, ga_string);
        serialize(n_trees, ga_string);
        serialize(init_batch_size, ga_string);
        serialize(bootstrap, ga_string);

        return ga_string;
    }

    void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            return model->fit_gd(X,Y,n_trees,bootstrap,init_batch_size,n_rounds,n_batches);
        } 
    }

    void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            return model->next_gd(X,Y,n_batches);
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
    
};

#endif