#pragma once

#include <vector>

#include "Datatypes.h"
#include "DecisionTree.h"
#include "TreeEnsemble.h"
#include "Utils.h"

template <typename data_t, INIT tree_init>
class OSE : public TreeEnsemble<data_t, tree_init> {
    
protected:
    
    unsigned int burnin_steps;
    unsigned int n_trees;
    unsigned int batch_size;
    bool bootstrap; 
    unsigned int epochs;

public:

    OSE(
        unsigned int n_classes, 
        unsigned int max_depth,
        unsigned int max_features,
        std::string loss = "mse",
        std::string optimizer = "adam",
        internal_t step_size = 0.1,
        unsigned long seed = 12345,
        bool normalize_weights = true,
        unsigned int burnin_steps = 0,
        std::string regularizer = "L0",
        internal_t l_reg = 0,
        unsigned int n_trees = 32,
        unsigned int batch_size = 0,
        unsigned int bootstrap = true,
        unsigned int epochs = 0
    ) : TreeEnsemble<data_t, tree_init>(n_classes, nullptr, nullptr, seed, false, nullptr, nullptr, std::nullopt, l_reg, std::nullopt, 0), burnin_steps(burnin_steps), n_trees(n_trees), batch_size(batch_size), bootstrap(bootstrap), epochs(epochs) {
        if (loss == "mse" || loss == "MSE") {
            this->loss = std::make_unique<MSE>();
        } else if (loss == "CrossEntropy" || loss == "CROSSENTROPY") {
            this->loss = std::make_unique<CrossEntropy>();
        } else {
            std::runtime_error("Received a parameter that I did not understand. I understand loss = {mse, CrossEntropy}, but you gave me " + loss);
        }

        if (optimizer == "adam" || optimizer == "ADAM") {
            this->weight_optimizer = std::make_unique<Adam>(step_size);
        } else if (optimizer == "sgd" || optimizer == "SGD") {
            this->weight_optimizer = std::make_unique<SGD>(step_size);
        } else {
            std::runtime_error("Received a parameter that I did not understand. I understand optimizer = {adam, sgd}, but you gave me " + optimizer);
        }

        if (regularizer == "l0" || regularizer == "L0") {
            this->ensemble_regulizer = L0_reg;
        } else if (regularizer == "l1" || regularizer == "L1") {
            this->ensemble_regulizer = L1_reg;
        } else if (regularizer == "l2" || regularizer == "L2") {
            this->ensemble_regulizer = L2_reg;
        } else {
            std::runtime_error("Received a parameter that I did not understand. I understand regularizer = {L0, L1, L2}, but you gave me " + regularizer);
        }
    }

    OSE(
        unsigned int n_classes, 
        unsigned int max_depth, 
        unsigned int max_features, 
        Loss const & loss,
        Optimizer const & optimizer,
        unsigned long seed = 12345,
        bool normalize_weights = true,
        unsigned int burnin_steps = 0,
        std::optional<std::function< std::vector<internal_t>(std::vector<internal_t> const &, internal_t scale)>> ensemble_regulizer = std::nullopt,
        internal_t l_reg = 0,
        unsigned int n_trees = 32,
        unsigned int batch_size = 0,
        unsigned int bootstrap = true,
        unsigned int epochs = 0
    ) : TreeEnsemble<data_t, tree_init>(n_classes, max_depth, max_features, nullptr, nullptr, seed, false, nullptr, nullptr, ensemble_regulizer, l_reg, std::nullopt, 0), burnin_steps(burnin_steps), n_trees(n_trees), batch_size(batch_size), bootstrap(bootstrap), epochs(epochs) {
        this->loss = loss.clone();
        this->weight_optimizer = optimizer.clone(); 
    }

    void next(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        this->_weights.push_back(0.0);

        if constexpr (tree_init == INIT::CUSTOM) {
            this->_trees.push_back(DecisionTree<data_t, tree_init>(this->n_classes, this->max_depth, this->max_features, this->seed++, this->score));
        } else {
            this->_trees.push_back(DecisionTree<data_t, tree_init>(this->n_classes, this->max_depth, this->max_features, this->seed++));
        }

        this->_trees.back()->fit(X,Y);
        
        this->update_trees(X, Y, burnin_steps, std::nullopt, std::nullopt, this->seed);
        if (this->ensemble_regularizer.has_value()) {
            this->prune();
        }
    }

    void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        this->init_trees(X, Y, n_trees, bootstrap, batch_size);

        for (unsigned int i = 0; i < epochs; ++i) {
            unsigned int cnt = 0;

            unsigned int batch_cnt = 0;
            while(cnt < X.rows) {
                auto cur_batch_size = std::min(static_cast<int>(X.rows - cnt), static_cast<int>(batch_size));
                if (cur_batch_size <= 0) break;

                auto idx = sample_indices(X.rows, batch_size, bootstrap, this->seed + i);
                this->_weights.push_back(0.0);
                if (this->score.has_value()) {
                    this->_trees.push_back(DecisionTree<data_t, tree_init>(this->n_classes, this->max_depth, this->max_features, this->seed++, this->score));
                } else {
                    this->_trees.push_back(DecisionTree<data_t, tree_init>(this->n_classes, this->max_depth, this->max_features, this->seed++));
                }
                this->_trees.back().fit(X,Y,idx);
                
                this->update_trees(X, Y, burnin_steps, std::nullopt, std::nullopt, this->seed);
                if (this->ensemble_regulizer.has_value()) {
                    this->prune();
                }
            }
        }
    }
};