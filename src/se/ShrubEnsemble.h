#ifndef SHRUB_ENSEMBLE_H
#define SHRUB_ENSEMBLE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>

#include "Datatypes.h"
#include "Tree.h"
#include "Losses.h"
#include "EnsembleRegularizer.h"
#include "TreeRegularizer.h"

/**
 * @brief  Scales the given matrix X in-place by the given factor s
 * @note   
 * @param  &X: The matrix
 * @param  s: The scaling factor
 * @retval None, the operation changes X in-place
 */
void scale(std::vector<std::vector<data_t>> &X, data_t s) {
    for (unsigned int j = 0; j < X.size(); ++j) {
        for (unsigned int k = 0; k < X[j].size(); ++k) {
            X[j][k] *= s;
        }
    }
}

/**
 * @brief  Computes of all values in the given matrix X.
 * @note   
 * @param  &X: The matrix
 * @retval The mean of all matrix entries. 
 */
template<typename T>
T mean_all_dim(std::vector<std::vector<T>> &X) {
    unsigned int n_first = X.size();
    unsigned int n_second = X[0].size();
    T mean = 0;

    for (unsigned int j = 0; j < n_first; ++j) {
        for (unsigned int k = 0; k < n_second; ++k) {
            mean += X[j][k];
        }
    }

    return mean / (n_first * n_second);
}

/**
 * @brief Computes thes weighted sum across the first dimension of the given tensor using the supplied weights
 * @note   
 * @param  &X: A (N,M,K) tensor
 * @param  &weights: A (N,) vector
 * @retval A (M,K) matrix stored as std::vector<std::vector<data_t>>
 */
std::vector<std::vector<data_t>> weighted_sum_first_dim(std::vector<std::vector<std::vector<data_t>>> &X, std::vector<data_t> const &weights) {
    unsigned int n_first = X.size();
    unsigned int n_second = X[0].size();
    unsigned int n_third = X[0][0].size();

    std::vector<std::vector<data_t>> XMean(n_second, std::vector<data_t> (n_third, 0));

    for (unsigned int i = 0; i < n_first; ++i) {
        for (unsigned int j = 0; j < n_second; ++j) {
            for (unsigned int k = 0; k < n_third; ++k) {
                XMean[j][k] += X[i][j][k] * weights[i];
            }
        }
    }

    return XMean;
}

/**
 * @brief  Samples `batch_size' data points from X and Y. If bootstrap is true, then sampling is performed with replacement. Otherwhise no replacement is used. This functions returns a tuple in which the first entry is the sampled data of type std::vector<std::vector<data_t>> and second entry is the sampled label of type  std::vector<unsigned int>
 * @note   
 * @param  &X: The (N,d) data matrix
 * @param  &Y: The (N,) label vector
 * @param  batch_size: The batch size
 * @param  bootstrap: If true, samples with replacement. If false, no replacement is used
 * @param  &gen: The random generator used for sampling
 * @retval A tuple in which the first entry is a (batch_size, d) matrix (stored as std::vector<std::vector<data_t>>) and the second entry is a (batch_size) vector (stored as std::vector<unsigned int>)
 */
auto sample_data(std::vector<std::vector<data_t>>const &X, std::vector<unsigned int>const &Y, unsigned int batch_size, bool bootstrap, std::minstd_rand &gen) {
    if (batch_size >= X.size() || batch_size == 0) {
        batch_size = X.size();
    }

    std::vector<std::vector<data_t>> bX(batch_size);
    std::vector<unsigned int> bY(batch_size);

    if (bootstrap) {
        std::uniform_int_distribution<> dist(0, X.size()-1); 
        for (unsigned int i = 0; i < batch_size; ++i) {
            auto idx = dist(gen);
            bX[i] = X[idx];
            bY[i] = Y[idx];
        }
    } else {
        std::vector<unsigned int> idx(X.size());
        std::iota(std::begin(idx), std::end(idx), 0);
        std::shuffle(idx.begin(), idx.end(), gen);
        for (unsigned int i = 0; i < batch_size; ++i) {
            bX[i] = X[idx[i]];
            bY[i] = Y[idx[i]];
        }
    }

    return std::make_tuple(bX, bY);
}

/**
 * @brief  Samples `batch_size' data points from X and Y. If bootstrap is true, then sampling is performed with replacement. Otherwhise no replacement is used. This functions returns a tuple in which the first entry is the sampled data of type std::vector<std::vector<data_t>> and second entry is the sampled label of type  std::vector<unsigned int>
 * @note   
 * @param  &X: The (N,d) data matrix
 * @param  &Y: The (N,) label vector
 * @param  batch_size: The batch size
 * @param  bootstrap: If true, samples with replacement. If false, no replacement is used
 * @param  seed: The random generator seed used for seeding a std::minstd_rand random generator. Defaults to 12345L.
 * @retval A tuple in which the first entry is a (batch_size, d) matrix (stored as std::vector<std::vector<data_t>>) and the second entry is a (batch_size) vector (stored as std::vector<unsigned int>)
 */
auto sample_data(std::vector<std::vector<data_t>>const &X, std::vector<unsigned int>const &Y, unsigned int batch_size, bool bootstrap, unsigned long seed = 12345L) {
    std::minstd_rand gen(seed);
    return sample_data(X,Y,batch_size,bootstrap,gen);
}

/**
 * @brief  
 * @note   
 * @retval None
 */
template <LOSS::TYPE loss_type, OPTIMIZER::OPTIMIZER_TYPE opt, OPTIMIZER::OPTIMIZER_TYPE tree_opt, TREE_INIT tree_init>
class ShrubEnsemble {

private:
    std::vector< Tree<tree_init, tree_opt> > _trees;
    std::vector<internal_t> _weights;

    unsigned int const n_classes;
    unsigned int const max_depth;
    unsigned long seed;

    bool const normalize_weights;
    unsigned int const burnin_steps;
    unsigned int const max_features;

    OPTIMIZER::Optimizer<opt, OPTIMIZER::STEP_SIZE_TYPE::CONSTANT> optimizer;
    internal_t const step_size;

    LOSS::Loss<loss_type> loss;
    // std::function< std::vector<std::vector<internal_t>>(std::vector<std::vector<internal_t>> const &, std::vector<unsigned int> const &) > loss;
    // std::function< std::vector<std::vector<internal_t>>(std::vector<std::vector<internal_t>> const &, std::vector<unsigned int> const &) > loss_deriv;

    std::function< std::vector<internal_t>(std::vector<internal_t> const &, data_t scale) > ensemble_regularizer;
    data_t const l_ensemble_reg;
    
    std::function< internal_t(Tree<tree_init, tree_opt> const &) > tree_regularizer;
    internal_t const l_tree_reg;

public:

    /**
     * @brief  Constructs a new ShrubEnsembles object.
     * @note   
     * @param  n_classes: The number of classes on which this object should be fitted. This is necessary for online learning in which new classes may arrive over time and hence the total number of classes must be given beforehand.
     * @param  max_depth: The maximum depth of the decision trees. If max_depth = 0 then trees are trained until leaf nodes are pure. Defaults to 5. 
     * @param  seed: The random seed used for all randomizations. Defaults to 12345. 
     * @param  normalize_weights: If true then the weights are normalized so that the sum to 1. Otherwise unnormalized weights are used. Defaults to true.
     * @param  burnin_steps: The number of `burn-in' steps performed by the algorithm. The number of burn-in steps are the number of SGD updates the algorithm performs for each call of the next() function. In case of distributed training burn-in refers to the number of SGD updates each worker performs individually before model averaging.  Defaults to 0. 
     * @param  max_features: The maximum number features used to fit the decision trees. If max_features = 0, then all features are used. Defaults to 0.
     * @param  loss: The loss function that is minimized by this algorithm. See LOSS::TYPE for available losses. Defaults to LOSS::TYPE::MSE.
     * @param  step_size: The step-size used for any of the SGD updates. 
     * @param  ensemble_regularizer: The ensemble regularizer that is added to the loss function. See ENSEMBLE_REGULARIZER::TYPE for available losses. Defaults to ENSEMBLE_REGULARIZER::TYPE::NO.
     * @param  l_ensemble_reg: The regularization strength for the ensemble_regularizer. Defaults to 0.
     * @param  tree_regularizer: The tree regularizer that is added to the loss function. See TREE_REGULARIZER::TYPE for available losses. Defaults to TREE_REGULARIZER::TYPE::NO.
     * @param  l_tree_reg: The regularization strength for the tree_regularizer. Defaults to 0.
     * @retval A new ShrubEnsembles object
     */
    ShrubEnsemble(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        bool normalize_weights = true,
        unsigned int burnin_steps = 0,
        unsigned int max_features = 0,
        // LOSS::TYPE loss = LOSS::TYPE::MSE,
        internal_t step_size = 1e-2,
        ENSEMBLE_REGULARIZER::TYPE ensemble_regularizer = ENSEMBLE_REGULARIZER::TYPE::NO,
        internal_t l_ensemble_reg = 0.0,
        TREE_REGULARIZER::TYPE tree_regularizer = TREE_REGULARIZER::TYPE::NO,
        internal_t l_tree_reg = 0.0
    ) : 
        n_classes(n_classes), 
        max_depth(max_depth), 
        seed(seed), 
        normalize_weights(normalize_weights), 
        burnin_steps(burnin_steps),
        max_features(max_features),
        optimizer(step_size),
        step_size(step_size),
        loss(),
        // loss(LOSS::from_enum(loss)), 
        // loss_deriv(LOSS::deriv_from_enum(loss)), 
        ensemble_regularizer(ENSEMBLE_REGULARIZER::from_enum(ensemble_regularizer)), 
        l_ensemble_reg(l_ensemble_reg),
        tree_regularizer(TREE_REGULARIZER::from_enum<tree_init, tree_opt>(tree_regularizer)),
        l_tree_reg(l_tree_reg)
    {}

    unsigned int num_bytes() const {
        unsigned int tree_size = 0;
        for (auto const & t : _trees) {
            tree_size += t.num_bytes();
        }

        return tree_size + sizeof(*this) + optimizer.num_bytes();
    }

    /**
     * @brief  Remove all trees including their weight which have a 0 weight. 
     * @note   
     * @retval None
     */
    void prune() {
        // Remove all trees and weights which have 0 weight
        unsigned int before = _weights.size();
        auto wit = _weights.begin();
        auto tit = _trees.begin();

        while (wit != _weights.end() && tit != _trees.end()) {
            if (*wit == 0) {
                wit = _weights.erase(wit);
                tit = _trees.erase(tit);
            } else {
                ++wit;
                ++tit;
            }
        }

        if (before != _weights.size()) {
            // Make sure to reset the optimizer (e.g. the m/v estimates in ADAM) if a weight has been removed
            // because they are now obsolet
            optimizer.reset();
        }
    }

    void init_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool boostrap, unsigned int batch_size) {
        
        // Create the tree objects and initialize their weight. 
        // We do this in a single thread so that we can perform the training without any
        // synchroization
        for (unsigned int i = 0; i < n_trees; ++i) {
            _trees.push_back(Tree<tree_init, tree_opt>(n_classes, max_depth, max_features, seed+i, step_size));
            _weights.push_back(1.0 / n_trees);    
        }
        // Make sure to advance the random seed "properly"
        seed += n_trees;

        // Do the training in parallel
        #pragma omp parallel for
        for (unsigned int i = 0; i < n_trees; ++i){
            auto s = sample_data(X, Y, batch_size, boostrap, seed + i);
            _trees[i].fit(std::get<0>(s), std::get<1>(s));
        }
        // Make sure to advance the random seed "properly"
        seed += n_trees;
    }

    void next_distributed(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_parallel, bool boostrap, unsigned int batch_size) {
        std::vector<ShrubEnsemble<loss_type, opt, tree_opt, tree_init>> ses(_trees.size(), *this);

        #pragma omp parallel for
        for (unsigned int k = 0; k < n_parallel; ++k){
            auto s = sample_data(X, Y, batch_size, boostrap, seed+k);
            ses[k].update_trees(std::get<0>(s), std::get<1>(s));
        }
        seed += n_parallel;

        #pragma omp parallel for 
        for (unsigned int j = 0; j < _trees.size(); ++j) {
            for (unsigned int k = 0; k < ses.size(); ++k) {
                if ( k == 0) {
                    _weights[j] = ses[k]._weights[j];
                    _trees[j].leafs = ses[k]._trees[j].leafs;
                } else {
                    _weights[j] += ses[k]._weights[j];

                    for (unsigned int l = 0; l < ses[k]._trees[j].leafs.size(); ++l) {
                        _trees[j].leafs[l] += ses[k]._trees[j].leafs[l];
                    }
                }
            }
            _weights[j] /= n_parallel;
            std::transform(_trees[j].leafs.begin(), _trees[j].leafs.end(), _trees[j].leafs.begin(), [n_parallel](auto& c){return 1.0/n_parallel*c;});
        }
    }

    void fit_distributed(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size, unsigned int n_rounds) {
        init_trees(X, Y, n_trees, bootstrap, batch_size);
        
        for (unsigned int i = 0; i < n_rounds; ++i) {
            next_distributed(X,Y,n_trees,bootstrap,batch_size);
            if constexpr (opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                // TODO also skip if ensemble_regularizer is NO
                prune();
            }
        }
    }

    void next_gd(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_batches) {
        if constexpr(tree_opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
            // Put all gradients in all_grad which is populated in parallel by n_batches threads
            std::vector<std::vector<std::vector<internal_t>>> all_grad(n_batches);

            if (X.size() < n_batches) {
                n_batches = X.size();
            }
            unsigned int b_size = X.size() / n_batches;

            // Compute the gradients in n_batches and store the aggregated gradients in all_grad for each batch
            // After that we average the gradients in all_grad and perform the GD update. 
            #pragma omp parallel for
            for (unsigned int b = 0; b < n_batches; ++b) {
                unsigned int actual_size = b_size;

                // The last thread works on all remaining data items if they are unevenly distributed.
                if (b == n_batches - 1) {
                    actual_size = X.size() - b_size * b;
                } 

                // Apply each tree and store the leaf index for each example in the current batch in idx. 
                // Compute the ensembles output and store it in output
                std::vector<std::vector<unsigned int>> idx(_trees.size(), std::vector<unsigned int>(actual_size));
                std::vector<std::vector<internal_t>> output(actual_size, std::vector<internal_t> (n_classes, 0));
                for (unsigned int i = 0; i < _trees.size(); ++i) {
                    // idx[i].reserve(actual_size);
                    for (unsigned int j = 0; j < actual_size; ++j) {
                        auto const & x = X[b*b_size + j];
                        auto lidx = _trees[i].leaf_index(x);
                        // idx[i][j].push_back(lidx);

                        idx[i][j] = lidx;
                        for (unsigned int k = 0; k < n_classes; ++k) {
                            output[j][k] += _weights[i] * _trees[i].leafs[lidx + k];
                        }
                    }
                }

                // Make sure we have enough space to access the gradients for the current batch 
                all_grad[b] = std::vector<std::vector<internal_t>>(_trees.size());
                std::vector<internal_t> loss_deriv(n_classes);

                for (unsigned int i = 0; i < _trees.size(); ++i) {
                    // Compute gradient for current tree
                    all_grad[b][i] = std::vector<internal_t>(_trees[i].leafs.size(), 0);
                    for (unsigned int k = 0; k < actual_size; ++k) {
                        // No need to reset loss_deriv because it will be copied anyway
                        auto y = Y[b*b_size + k];
                        loss.deriv(output[k], loss_deriv, y, n_classes);

                        auto lidx = idx[i][k];
                        for (unsigned int j = 0; j < n_classes; ++j) {
                            all_grad[b][i][lidx+j] += loss_deriv[j] * _weights[i] * 1.0 / actual_size * 1.0 / n_classes;
                        }
                        // TODO use transform here?
                    }
                }
            }

            // All aggregated gradients are now stored in all_grad
            // Now perform the update for each tree. 
            #pragma omp parallel for
            for (unsigned int j = 0; j < _trees.size(); ++j) {
                std::vector<internal_t> t_grad(_trees[j].leafs.size(), 0); 
                for (unsigned int i = 0; i < n_batches; ++i) {
                    for (unsigned int l = 0; l < t_grad.size(); ++l) {
                        t_grad[l] += all_grad[i][j][l];
                    }
                }
                std::transform(t_grad.begin(), t_grad.end(), t_grad.begin(), [n_batches](auto& c){return 1.0/n_batches*c;});
                _trees[j].optimizer.step(_trees[j].leafs, t_grad);
            }
        }
    }

    void fit_gd(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size, unsigned int n_rounds, unsigned int n_batches) {
        init_trees(X, Y, n_trees, bootstrap, batch_size);
        
        for (unsigned int i = 0; i < n_rounds; ++i) {
            auto batch = sample_data(X,Y,batch_size,bootstrap,seed++);

            next_gd(X,Y,n_batches);
            if constexpr (opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                // TODO also skip if ensemble_regularizer is NO
                prune();
            }
        }
    }

    void update_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        // The structure of the trees does not change with the optimization and hence we can 
        // pre-compute the leaf indices for each tree / sample and store them. This mitigates the
        // somewhat "costly" iteration of the trees in each round but gives direct access to the
        // leaf nodes
        std::vector<std::vector<unsigned int>> idx(_trees.size(), std::vector<unsigned int>(X.size()));
        
        #pragma omp parallel for
        for (unsigned int i = 0; i < _trees.size(); ++i) {
            for (unsigned int j = 0; j < X.size(); ++j) {
                idx[i][j] = _trees[i].leaf_index(X[j]);
            }
        }

        // Store the current predictions in the output vector. 
        std::vector<std::vector<internal_t>> output(X.size(), std::vector<internal_t> (n_classes, 0));
        for (unsigned int s = 0; s < burnin_steps + 1; ++s) {
            
            // Reset the output vector because we "add into" it in the for loop below
            // Compute the predictions for each tree / sample with the pre-computed indices.
            // This can be done a bit more efficient if we would update the output vector after the gradient step
            // instead of recomputing the entire predictions from scratch. But this is more readable and 
            // maintainable
            for(auto & o : output) {
                std::fill(o.begin(), o.end(), static_cast<internal_t>(0));
            }

            #pragma omp parallel for
            for (unsigned int i = 0; i < _trees.size(); ++i) {
                for (unsigned int j = 0; j < X.size(); ++j) {
                    auto lidx = idx[i][j];

                    for (unsigned int k = 0; k < n_classes; ++k) {
                        output[j][k] += _weights[i] * _trees[i].leafs[lidx + k];
                    }
                }
            }

            if constexpr(tree_opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                
                #pragma omp parallel for
                for (unsigned int i = 0; i < _weights.size(); ++i) {
                    std::vector<internal_t> loss_deriv(n_classes, 0);
                    std::vector<internal_t> grad(_trees[i].leafs.size(), 0);

                    // Compute gradient for current tree
                    for (unsigned int k = 0; k < X.size(); ++k) {
                        
                        // No need to reset loss_deriv because it will be copied anyway
                        loss.deriv(&output[k][0], &loss_deriv[0], Y[k], n_classes);

                        auto lidx = idx[i][k];
                        for (unsigned int j = 0; j < n_classes; ++j) {
                            grad[lidx+j] += loss_deriv[j] * _weights[i] * 1.0 / X.size() * 1.0 / n_classes;
                        }
                    }
                    // Update current tree
                    _trees[i].optimizer.step(_trees[i].leafs, grad);
                }
            }

            if constexpr(opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                // Compute gradient for the weights
                std::vector<internal_t> grad(_weights.size(), 0);

                #pragma omp parallel for
                for (unsigned int i = 0; i < _trees.size(); ++i) {
                    std::vector<internal_t> loss_deriv(n_classes, 0);
                    internal_t dir = 0;

                    // Compute tree regularization if necessary
                    if (l_tree_reg > 0) {
                        dir += l_tree_reg * tree_regularizer(_trees[i]);
                    }

                    // Compute gradient for tree i
                    for (unsigned int j = 0; j < X.size(); ++j) {
                        loss.deriv(&output[j][0], &loss_deriv[0], Y[j], n_classes);

                        auto lidx = idx[i][j];
                        for (unsigned int k = 0; k < n_classes; ++k) {
                            dir += _trees[i].leafs[lidx + k] * loss_deriv[k];
                        }
                    }
                    grad[i] = dir / (X.size() * n_classes);
                }

                // Perform SGD step for weights and apply prox operator afterwards
                optimizer.step(_weights, grad);
                _weights = ensemble_regularizer(_weights, l_ensemble_reg);
            
                if (normalize_weights && _weights.size() > 0) {
                    std::vector<internal_t> nonzero_w;
                    std::vector<unsigned int> nonzero_idx;
                    for (unsigned int i = 0; i < _weights.size(); ++i) {
                        if (_weights[i] != 0) {
                            nonzero_w.push_back(_weights[i]);
                            nonzero_idx.push_back(i);
                        }
                    }
                    nonzero_w = ENSEMBLE_REGULARIZER::to_prob_simplex(nonzero_w);
                    for (unsigned int i = 0; i < nonzero_idx.size(); ++i) {
                        unsigned int idx = nonzero_idx[i];
                        _weights[idx] = nonzero_w[i];
                    }
                }
            }
        }
    }

    void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        _weights.push_back(0.0);
        _trees.push_back(Tree<tree_init, tree_opt>(n_classes,max_depth, max_features, seed++, step_size));
        _trees.back().fit(X,Y);
        
        update_trees(X, Y);
        if constexpr (opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
            // TODO also skip if ensemble_regularizer is NO
            prune();
        }
    }

    std::vector<std::vector<internal_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        std::vector<std::vector<internal_t>> output; 
        if (_trees.size() == 0) {
            output.resize(X.size());
            for (unsigned int i = 0; i < X.size(); ++i) {
                output[i] = std::vector<internal_t>(n_classes, 1.0/n_classes);
            }
        } else {
            std::vector<std::vector<std::vector<internal_t>>> all_proba(_trees.size());
            for (unsigned int i = 0; i < _trees.size(); ++i) {
                all_proba[i] = _trees[i].predict_proba(X);
            }
            output = weighted_sum_first_dim(all_proba, _weights);
        }
        return output;
    }

    unsigned int num_nodes() const {
        unsigned int n_nodes = 0;
        for (auto const & t : _trees) {
            n_nodes += t.num_nodes();
        }
        return n_nodes;
    }

    unsigned int num_trees() const {
        return _trees.size();
    }

    std::vector<internal_t> weights() const {
        return _weights;
    }
};

#endif