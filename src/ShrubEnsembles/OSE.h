#pragma once

#include <vector>

#include "Datatypes.h"
#include "DecisionTree.h"
#include "TreeEnsemble.h"
#include "Utils.h"

template <typename data_t>
class OSE : public TreeEnsemble<data_t> {
    
protected:
    
    unsigned int burnin_steps;
    unsigned int n_trees;
    unsigned int batch_size;
    bool bootstrap; 
    unsigned int epochs;

public:

    OSE(
        unsigned int n_classes, 
        Tree<data_t> const & tree,
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
    ) : TreeEnsemble<data_t>(n_classes, nullptr, nullptr, seed, false, nullptr, nullptr, ensemble_regulizer, l_reg, std::nullopt, 0), burnin_steps(burnin_steps), n_trees(n_trees), batch_size(batch_size), bootstrap(bootstrap), epochs(epochs) {
        this->the_tree = tree.clone(seed);
        this->loss = loss.clone();
        this->weight_optimizer = optimizer.clone(); 
    }

    void next(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        this->_weights.push_back(0.0);
        this->_trees.push_back(this->the_tree->clone(this->seed++));
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
                this->_trees.push_back(this->prototype->clone(this->seed++));
                this->_trees.back().fit(X,Y,idx);
                
                this->update_trees(X, Y, burnin_steps, std::nullopt, std::nullopt, this->seed);
                if (this->ensemble_regulizer.has_value()) {
                    this->prune();
                }
            }
        }
    }
};