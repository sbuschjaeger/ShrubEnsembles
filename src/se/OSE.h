#pragma once

#include <vector>

#include "Datatypes.h"
#include "DecisionTree.h"
#include "TreeEnsemble.h"
#include "EnsembleRegularizer.h"
#include "TreeRegularizer.h"
#include "Utils.h"

template <LOSS::TYPE loss_type, OPTIMIZER::OPTIMIZER_TYPE opt, OPTIMIZER::OPTIMIZER_TYPE tree_opt, ENSEMBLE_REGULARIZER::TYPE ensemble_reg, DT::TREE_INIT tree_init>
class OSE : public TreeEnsemble<loss_type, opt, tree_opt, tree_init> {
    
protected:
    
    unsigned int burnin_steps;
    unsigned int n_trees;
    unsigned int batch_size;
    bool bootstrap; 
    unsigned int epochs;

public:

    OSE(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        bool normalize_weights = true,
        unsigned int burnin_steps = 0,
        unsigned int max_features = 0,
        internal_t step_size = 1e-2,
        internal_t l_reg = 0,
        unsigned int n_trees = 32,
        unsigned int batch_size = 0,
        unsigned int bootstrap = true,
        unsigned int epochs = 0
    ) : TreeEnsemble<loss_type, opt, tree_opt, tree_init>(n_classes, max_depth, seed, false, max_features, step_size, ensemble_reg, l_reg, TREE_REGULARIZER::TYPE::NO, 0), burnin_steps(burnin_steps), n_trees(n_trees), batch_size(batch_size), bootstrap(bootstrap), epochs(epochs) { 
    }

    void next(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        this->_weights.push_back(0.0);
        this->_trees.push_back(DecisionTree<tree_init, tree_opt>(this->n_classes, this->max_depth, this->max_features, this->seed++, this->step_size));
        this->_trees.back().fit(X,Y);
        
        this->update_trees(X, Y, burnin_steps, std::nullopt, std::nullopt, this->seed);
        if constexpr (opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
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
                this->_trees.push_back(DecisionTree<tree_init, tree_opt>(this->n_classes, this->max_depth, this->max_features, this->seed++, this->step_size));
                this->_trees.back().fit(X,Y,idx);
                
                this->update_trees(X, Y, burnin_steps, std::nullopt, std::nullopt, this->seed);
                if constexpr (opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                    this->prune();
                }
            }
        }
    }
};