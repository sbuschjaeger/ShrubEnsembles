#pragma once

#include <vector>

#include "Datatypes.h"
#include "Matrix.h"
#include "DecisionTree.h"
#include "TreeEnsemble.h"
#include "EnsembleRegularizer.h"
#include "TreeRegularizer.h"

template <LOSS::TYPE loss_type, OPTIMIZER::OPTIMIZER_TYPE opt, OPTIMIZER::OPTIMIZER_TYPE tree_opt>
class MASE : public TreeEnsemble<loss_type, opt, tree_opt> {
    
protected:
    unsigned int const n_trees;
    unsigned int const n_rounds;
    unsigned int const batch_size;
    bool const bootstrap; 
    unsigned int const burnin_steps;
    unsigned int const n_worker;
    unsigned int const init_tree_size;

public:

    MASE(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        unsigned int burnin_steps = 0,
        unsigned int max_features = 0,
        internal_t step_size = 1e-2,
        unsigned int n_trees = 32, 
        unsigned int n_worker = 8, 
        unsigned int n_rounds = 5,
        unsigned int batch_size = 0,
        bool bootstrap = true, 
        unsigned int init_tree_size = 0
    ) : TreeEnsemble<loss_type, opt, tree_opt>(n_classes, max_depth, seed, false, max_features, step_size, ENSEMBLE_REGULARIZER::TYPE::NO, 0, TREE_REGULARIZER::TYPE::NO, 0), n_trees(n_trees), n_rounds(n_rounds), batch_size(batch_size), bootstrap(bootstrap), burnin_steps(burnin_steps), n_worker(n_worker), init_tree_size(init_tree_size) { 
    }

    void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, unsigned int n_trees, unsigned int init_tree_size, unsigned int n_parallel, bool bootstrap, unsigned int batch_size, unsigned int n_rounds, unsigned int burnin_steps) {
        this->init_trees(X, Y, n_trees, bootstrap, init_tree_size);
        
        for (unsigned int i = 0; i < n_rounds; ++i) {
            next(X,Y,n_parallel,bootstrap,batch_size,burnin_steps);
            if constexpr (opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                this->prune();
            }
        }
    }

    void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        fit(X, Y, n_trees, init_tree_size, n_worker, bootstrap, batch_size, n_rounds, burnin_steps);
    }

    void next(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        if (n_worker == 1) {
            this->update_trees(X, Y, burnin_steps,batch_size, bootstrap);
        } else {
            next(X,Y,n_worker,bootstrap, batch_size, burnin_steps);
        }
    }

    void next(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, unsigned int n_parallel, bool boostrap, unsigned int batch_size, unsigned int burnin_steps) {
        std::vector<TreeEnsemble<loss_type, opt, tree_opt>> ses(n_parallel, *this);

        #pragma omp parallel for
        for (unsigned int k = 0; k < n_parallel; ++k){
            ses[k].update_trees(X,Y,burnin_steps,batch_size,boostrap,this->seed+k);
        }
        this->seed += n_parallel;

        #pragma omp parallel for 
        for (unsigned int j = 0; j < this->_trees.size(); ++j) {
            for (unsigned int k = 0; k < ses.size(); ++k) {
                if ( k == 0) {
                    this->_weights[j] = ses[k]._weights[j];
                    this->_trees[j]._leaves = ses[k]._trees[j]._leaves;
                } else {
                    this->_weights[j] += ses[k]._weights[j];

                    for (unsigned int l = 0; l < ses[k]._trees[j]._leaves.size(); ++l) {
                       this-> _trees[j]._leaves[l] += ses[k]._trees[j]._leaves[l];
                    }
                }
            }
            this->_weights[j] /= n_parallel;
            std::transform(this->_trees[j]._leaves.begin(), this->_trees[j]._leaves.end(), this->_trees[j]._leaves.begin(), [n_parallel](auto& c){return 1.0/n_parallel*c;});
        }
    }

    void init(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        if (init_tree_size == 0 || init_tree_size > X.rows) {
            this->init(X,Y,n_trees,bootstrap,X.rows);
        } else {
            this->init(X,Y,n_trees,bootstrap,init_tree_size);
        }
    }
};