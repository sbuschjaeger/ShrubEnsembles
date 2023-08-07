#pragma once

#include <vector>

#include "Datatypes.h"
#include "DecisionTree.h"
#include "TreeEnsemble.h"
#include "EnsembleRegularizer.h"
#include "TreeRegularizer.h"

template <LOSS::TYPE loss_type, OPTIMIZER::OPTIMIZER_TYPE opt, OPTIMIZER::OPTIMIZER_TYPE tree_opt, DT::TREE_INIT tree_init>
class GASE : public TreeEnsemble<loss_type, opt, tree_opt, tree_init> {
    
private:
    unsigned int const n_rounds;
    unsigned int const n_worker;
    unsigned int const n_trees;
    unsigned int const init_batch_size;
    unsigned int const batch_size;
    bool const bootstrap; 

public:

    GASE(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        unsigned int max_features = 0,
        internal_t step_size = 1e-2,
        unsigned int n_trees = 32, 
        unsigned int n_worker = 5,
        unsigned int n_rounds = 5,
        unsigned int init_batch_size = 0,
        unsigned int batch_size = 0,
        bool bootstrap = true
    ) : TreeEnsemble<loss_type, opt, tree_opt, tree_init>(n_classes, max_depth, seed, false, max_features, step_size, ENSEMBLE_REGULARIZER::TYPE::NO, 0, TREE_REGULARIZER::TYPE::NO, 0), n_rounds(n_rounds), n_worker(n_worker), n_trees(n_trees), init_batch_size(init_batch_size), batch_size(batch_size), bootstrap(bootstrap) { 
        
    }
    
    void next(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, unsigned int n_worker, unsigned int batch_size) {
        if constexpr(tree_opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
            // Put all gradients in all_grad which is populated in parallel by n_worker threads

            // TODO Remove std::vectors as much as possibru
            std::vector<std::vector<std::vector<internal_t>>> all_grad(n_worker);

            if (X.rows < n_worker) {
                n_worker = X.rows;
            }
            
            std::minstd_rand gen(this->seed++);
            std::vector<unsigned int> sample_idx(X.rows);
            std::iota(sample_idx.begin(), sample_idx.end(), 0);
            std::shuffle(sample_idx.begin(), sample_idx.end(), gen);

            unsigned int b_size = batch_size; 
            if (batch_size == 0 || batch_size*n_worker > X.rows) {
                b_size = static_cast<unsigned int>(X.rows / n_worker);
            }

            // Compute the gradients in n_worker and store the aggregated gradients in all_grad for each batch
            // After that we average the gradients in all_grad and perform the GD update. 
            #pragma omp parallel for
            for (unsigned int b = 0; b < n_worker; ++b) {
                unsigned int actual_size = b_size;

                // The last thread works on all remaining data items if they are unevenly distributed.
                if (batch_size == 0 && b == n_worker - 1) {
                    actual_size = X.rows - b_size * b;
                } 

                // Apply each tree and store the leaf index for each example in the current batch in idx. 
                // Compute the ensembles output and store it in output

                matrix2d<unsigned int> idx(this->_trees.size(), actual_size);
                matrix2d<internal_t> output(actual_size, this->n_classes);
                std::fill(output.begin(), output.end(), 0);

                for (unsigned int i = 0; i < this->_trees.size(); ++i) {
                    // idx[i].reserve(actual_size);
                    for (unsigned int j = 0; j < actual_size; ++j) {
                        auto xidx = sample_idx[b*b_size + j];
                        auto const & x = X(xidx);

                        auto lidx = this->_trees[i].leaf_index(x);
                        // idx[i][j].push_back(lidx);

                        idx(i,j) = lidx;
                        for (unsigned int k = 0; k < this->n_classes; ++k) {
                            output(j,k) += this->_weights[i] * this->_trees[i]._leaves[lidx + k];
                        }
                    }
                }

                // Make sure we have enough space to access the gradients for the current batch 
                all_grad[b] = std::vector<std::vector<internal_t>>(this->_trees.size());
                matrix1d<internal_t> loss_deriv(this->n_classes);

                for (unsigned int i = 0; i < this->_trees.size(); ++i) {
                    // Compute gradient for current tree
                    all_grad[b][i] = std::vector<internal_t>(this->_trees[i]._leaves.size(), 0);
                    for (unsigned int k = 0; k < actual_size; ++k) {
                        // No need to reset loss_deriv because it will be copied anyway
                        auto yidx = sample_idx[b*b_size + k];
                        auto y = Y(yidx);
                        this->loss.deriv(output(k), loss_deriv, y);

                        auto lidx = idx(i,k);
                        for (unsigned int j = 0; j < this->n_classes; ++j) {
                            all_grad[b][i][lidx+j] += loss_deriv(j) * this->_weights[i] * 1.0 / actual_size * 1.0 / this->n_classes;
                        }
                        // TODO use transform here?
                    }
                }
            }

            // All aggregated gradients are now stored in all_grad
            // Now perform the update for each tree. 
            #pragma omp parallel for
            for (unsigned int j = 0; j < this->_trees.size(); ++j) {
                matrix1d<internal_t> t_grad(this->_trees[j]._leaves.size());
                std::fill(t_grad.begin(), t_grad.end(), 0);

                for (unsigned int i = 0; i < n_worker; ++i) {
                    for (unsigned int l = 0; l < t_grad.dim; ++l) {
                        t_grad(l) += all_grad[i][j][l];
                    }
                }
                std::transform(t_grad.begin(), t_grad.end(), t_grad.begin(), [n_worker](auto& c){return 1.0/n_worker*c;});
                matrix1d<internal_t> tleafs(this->_trees[j]._leaves.size(), &this->_trees[j]._leaves[0], false);
                this->_trees[j].optimizer.step(tleafs, t_grad);
            }
        }
    }

    void next(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        next(X,Y,n_worker,batch_size);
    }

    void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int init_batch_size, unsigned int batch_size, unsigned int n_rounds, unsigned int n_worker) {
        this->init_trees(X, Y, n_trees, bootstrap, init_batch_size);
        
        for (unsigned int i = 0; i < n_rounds; ++i) {
            // TODO add sampling here? 
            this->next(X,Y,n_worker,batch_size);
            if constexpr (opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                // TODO also skip if ensemble_regularizer is NO
                this->prune();
            }
        }
    }

    void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        fit(X,Y,n_trees,bootstrap,init_batch_size,batch_size,n_rounds,n_worker);
    }

    void init(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        init(X,Y,n_trees,bootstrap,init_batch_size);
    }
};