#ifndef SHRUB_ENSEMBLE_H
#define SHRUB_ENSEMBLE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>
#include <optional>

#include "Datatypes.h"
#include "DecisionTree.h"
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

std::vector<unsigned int> sample_indices(unsigned int n_data, unsigned int batch_size, bool bootstrap, std::minstd_rand &gen) {
    if (batch_size >= n_data || batch_size == 0) {
        batch_size = n_data;
    }
    std::vector<unsigned int> idx(batch_size);
    if (bootstrap) {
        std::uniform_int_distribution<> dist(0, n_data - 1); 
        for (unsigned int i = 0; i < batch_size; ++i) {
            idx[i] = dist(gen);
        }
    } else {
        std::vector<unsigned int> _idx(n_data);
        std::iota(std::begin(_idx), std::end(_idx), 0);
        std::shuffle(_idx.begin(), _idx.end(), gen);

        for (unsigned int i = 0; i < batch_size; ++i) {
            idx[i] = _idx[i];
        }
    }
    return idx;
}

std::vector<unsigned int> sample_indices(std::vector<unsigned int> const &idx, unsigned int batch_size, bool bootstrap, std::minstd_rand &gen) {
    unsigned int n_data = idx.size();
    if (batch_size >= n_data || batch_size == 0) {
        batch_size = n_data;
    }
    std::vector<unsigned int> new_idx(batch_size);
    if (bootstrap) {
        std::uniform_int_distribution<> dist(0, n_data - 1); 
        for (unsigned int i = 0; i < batch_size; ++i) {
            new_idx[i] = idx[dist(gen)];
        }
    } else {
        std::vector<unsigned int> _idx(n_data);
        std::iota(std::begin(_idx), std::end(_idx), 0);
        std::shuffle(_idx.begin(), _idx.end(), gen);

        for (unsigned int i = 0; i < batch_size; ++i) {
            new_idx[i] = idx[_idx[i]];
        }
    }
    return new_idx;
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

std::vector<unsigned int> sample_indices(unsigned int n_data, unsigned int batch_size, bool bootstrap, unsigned long seed = 12345L) {
    std::minstd_rand gen(seed);
    return sample_indices(n_data,batch_size,bootstrap,gen);
}

std::vector<unsigned int> sample_indices(std::vector<unsigned int> const &idx, unsigned int batch_size, bool bootstrap, unsigned long seed = 12345L) {
    std::minstd_rand gen(seed);
    return sample_indices(idx,batch_size,bootstrap,gen);
}


class TreeEnsemble {
public:
    virtual void next_ma(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_parallel, bool boostrap, unsigned int batch_size, unsigned int burnin_steps) = 0;

    virtual void fit_ma(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, unsigned int n_parallel, bool bootstrap, unsigned int batch_size, unsigned int n_rounds, unsigned int burnin_steps) = 0;

    virtual void next_ga(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_batches) = 0;

    virtual void fit_ga(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size, unsigned int n_rounds, unsigned int n_batches) = 0;

    virtual void update_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int burnin_step = 1, std::optional<unsigned int> const batch_size = std::nullopt, std::optional<bool> bootstrap = std::nullopt,std::optional<unsigned long> seed = std::nullopt) = 0;

    // virtual void update_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y,  unsigned int burnin_step, std::vector<unsigned int> &data_idx) = 0;
    
    virtual void init_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool boostrap, unsigned int batch_size) = 0;
    
    virtual void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int burnin_steps) = 0;

    virtual unsigned int num_nodes() const = 0;

    virtual unsigned int num_bytes() const = 0;

    virtual unsigned int num_trees() const = 0;

    // virtual std::vector<internal_t> & weights() = 0;

    // virtual std::vector<Tree*> trees() = 0;
        
    virtual void load(std::vector<std::vector<internal_t>> & new_nodes, std::vector<std::vector<internal_t>> & new_leafs, std::vector<internal_t> & new_weights) = 0;

    virtual std::tuple<std::vector<std::vector<internal_t>>, std::vector<std::vector<internal_t>>, std::vector<internal_t>> store() const = 0;

    virtual std::vector<std::vector<internal_t>> predict_proba(std::vector<std::vector<data_t>> const &X) = 0;

    virtual ~TreeEnsemble() { }
};

/**
 * @brief  
 * @note   
 * @retval None
 */
template <LOSS::TYPE loss_type, OPTIMIZER::OPTIMIZER_TYPE opt, OPTIMIZER::OPTIMIZER_TYPE tree_opt, DT::TREE_INIT tree_init>
class ShrubEnsemble : public TreeEnsemble {

private:
    std::vector< DecisionTree<tree_init, tree_opt> > _trees;
    std::vector<internal_t> _weights;

    unsigned int const n_classes;
    unsigned int const max_depth;
    unsigned long seed;

    bool const normalize_weights;
    unsigned int const max_features;

    OPTIMIZER::Optimizer<opt, OPTIMIZER::STEP_SIZE_TYPE::CONSTANT> optimizer;
    internal_t const step_size;

    LOSS::Loss<loss_type> loss;
    // std::function< std::vector<std::vector<internal_t>>(std::vector<std::vector<internal_t>> const &, std::vector<unsigned int> const &) > loss;
    // std::function< std::vector<std::vector<internal_t>>(std::vector<std::vector<internal_t>> const &, std::vector<unsigned int> const &) > loss_deriv;

    std::function< std::vector<internal_t>(std::vector<internal_t> const &, data_t scale) > ensemble_regularizer;
    internal_t const l_ensemble_reg;
    
    std::function< internal_t(DecisionTree<tree_init, tree_opt> const &) > tree_regularizer;
    internal_t const l_tree_reg;

public:

    /**
     * @brief  Constructs a new ShrubEnsembles object.
     * @note   
     * @param  n_classes: The number of classes on which this object should be fitted. This is necessary for online learning in which new classes may arrive over time and hence the total number of classes must be given beforehand.
     * @param  max_depth: The maximum depth of the decision trees. If max_depth = 0 then trees are trained until leaf nodes are pure. Defaults to 5. 
     * @param  seed: The random seed used for all randomizations. Defaults to 12345. 
     * @param  normalize_weights: If true then the weights are normalized so that the sum to 1. Otherwise unnormalized weights are used. Defaults to true.
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
            _trees.push_back(DecisionTree<tree_init, tree_opt>(n_classes, max_depth, max_features, seed+i, step_size));
            _weights.push_back(1.0 / n_trees);    
        }
        // Make sure to advance the random seed "properly"
        seed += n_trees;

        // Do the training in parallel
        #pragma omp parallel for
        for (unsigned int i = 0; i < n_trees; ++i){
            auto idx = sample_indices(X.size(), batch_size, boostrap, seed + i);
            //auto s = sample_data(X, Y, batch_size, boostrap, seed + i);
            // _trees[i].fit(std::get<0>(s), std::get<1>(s));
            _trees[i].fit(X,Y,idx);

        }
        // Make sure to advance the random seed "properly"
        seed += n_trees;
    }

    void next_ma(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_parallel, bool boostrap, unsigned int batch_size, unsigned int burnin_steps) {
        std::vector<ShrubEnsemble<loss_type, opt, tree_opt, tree_init>> ses(n_parallel, *this);

        #pragma omp parallel for
        for (unsigned int k = 0; k < n_parallel; ++k){
            //auto idx = sample_indices(X.size(), batch_size, boostrap, seed + k);
            // auto s = sample_data(X, Y, batch_size, boostrap, seed+k);
            ses[k].update_trees(X,Y,burnin_steps,batch_size,boostrap,seed+k);
        }
        seed += n_parallel;

        #pragma omp parallel for 
        for (unsigned int j = 0; j < _trees.size(); ++j) {
            for (unsigned int k = 0; k < ses.size(); ++k) {
                if ( k == 0) {
                    _weights[j] = ses[k]._weights[j];
                    _trees[j]._leafs = ses[k]._trees[j]._leafs;
                } else {
                    _weights[j] += ses[k]._weights[j];

                    for (unsigned int l = 0; l < ses[k]._trees[j]._leafs.size(); ++l) {
                        _trees[j]._leafs[l] += ses[k]._trees[j]._leafs[l];
                    }
                }
            }
            _weights[j] /= n_parallel;
            std::transform(_trees[j]._leafs.begin(), _trees[j]._leafs.end(), _trees[j]._leafs.begin(), [n_parallel](auto& c){return 1.0/n_parallel*c;});
        }
    }

    void fit_ma(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, unsigned int n_parallel, bool bootstrap, unsigned int batch_size, unsigned int n_rounds, unsigned int burnin_steps) {
        init_trees(X, Y, n_trees, bootstrap, batch_size);
        
        for (unsigned int i = 0; i < n_rounds; ++i) {
            next_ma(X,Y,n_parallel,bootstrap,batch_size,burnin_steps);
            if constexpr (opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                // TODO also skip if ensemble_regularizer is NO
                prune();
            }
        }
    }

    void next_ga(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_worker) {
        if constexpr(tree_opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
            // Put all gradients in all_grad which is populated in parallel by n_worker threads
            std::vector<std::vector<std::vector<internal_t>>> all_grad(n_worker);

            if (X.size() < n_worker) {
                n_worker = X.size();
            }
            unsigned int b_size = X.size() / n_worker;

            // Compute the gradients in n_worker and store the aggregated gradients in all_grad for each batch
            // After that we average the gradients in all_grad and perform the GD update. 
            #pragma omp parallel for
            for (unsigned int b = 0; b < n_worker; ++b) {
                unsigned int actual_size = b_size;

                // The last thread works on all remaining data items if they are unevenly distributed.
                if (b == n_worker - 1) {
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
                            output[j][k] += _weights[i] * _trees[i]._leafs[lidx + k];
                        }
                    }
                }

                // Make sure we have enough space to access the gradients for the current batch 
                all_grad[b] = std::vector<std::vector<internal_t>>(_trees.size());
                std::vector<internal_t> loss_deriv(n_classes);

                for (unsigned int i = 0; i < _trees.size(); ++i) {
                    // Compute gradient for current tree
                    all_grad[b][i] = std::vector<internal_t>(_trees[i]._leafs.size(), 0);
                    for (unsigned int k = 0; k < actual_size; ++k) {
                        // No need to reset loss_deriv because it will be copied anyway
                        auto y = Y[b*b_size + k];
                        loss.deriv(&output[k][0], &loss_deriv[0], y, n_classes);

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
                std::vector<internal_t> t_grad(_trees[j]._leafs.size(), 0); 
                for (unsigned int i = 0; i < n_worker; ++i) {
                    for (unsigned int l = 0; l < t_grad.size(); ++l) {
                        t_grad[l] += all_grad[i][j][l];
                    }
                }
                std::transform(t_grad.begin(), t_grad.end(), t_grad.begin(), [n_worker](auto& c){return 1.0/n_worker*c;});
                _trees[j].optimizer.step(_trees[j]._leafs, t_grad);
            }
        }
    }

    void fit_ga(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size, unsigned int n_rounds, unsigned int n_worker) {
        init_trees(X, Y, n_trees, bootstrap, batch_size);
        
        for (unsigned int i = 0; i < n_rounds; ++i) {
            // auto batch = sample_data(X,Y,batch_size,bootstrap,seed++);

            // TODO add sampling here? 
            next_ga(X,Y,n_worker);
            if constexpr (opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                // TODO also skip if ensemble_regularizer is NO
                prune();
            }
        }
    }

    // void update_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int burnin_step = 1, std::optional<unsigned int> const batch_size = std::nullopt, std::optional<bool> bootstrap = std::nullopt) {
    //     //std::vector<unsigned int> idx(X.size());
    //     // TODO idx is not required here and takes some space. Can be optimized away
    //     //std::iota(std::begin(idx), std::end(idx), 0);
    //     this->update_trees(X,Y,burnin_step,batch_size,bootstrap); 
    // }

    void update_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int burnin_steps, std::optional<unsigned int> const batch_size = std::nullopt, std::optional<bool> bootstrap = std::nullopt,std::optional<unsigned long> seed = std::nullopt) {
        // The structure of the trees does not change with the optimization and hence we can 
        // pre-compute the leaf indices for each tree / sample and store them. This mitigates the
        // somewhat "costly" iteration of the trees in each round but gives direct access to the
        // leaf nodes

        unsigned int n_batch_size = X.size();
        if (batch_size.has_value() && batch_size.value() > 0) {
            n_batch_size = batch_size.value();
        }

        bool do_boostrap = false;
        if (bootstrap.has_value()) {
            do_boostrap = bootstrap.value();
        }

        unsigned long the_seed = 12345L;
        if (seed.has_value()) {
            the_seed = seed.value();
        }

        std::minstd_rand gen(the_seed);
        // unsigned int n_data = data_idx.size();
        // std::vector<std::vector<unsigned int>> leaf_idx(_trees.size(), std::vector<unsigned int>(n_data));
        
        // #pragma omp parallel for
        // for (unsigned int i = 0; i < _trees.size(); ++i) {
        //     for (unsigned int j = 0; j < n_data; ++j) {
        //         leaf_idx[i][j] = _trees[i].leaf_index(X[data_idx[j]]);
        //     }
        // }

        // Store the current predictions in the output vector. 
        std::vector<std::vector<internal_t>> output(n_batch_size, std::vector<internal_t> (n_classes, 0));
        for (unsigned int s = 0; s < burnin_steps + 1; ++s) {
            //auto idx = sample_indices(data_idx, n_batch_size, do_boostrap);
            auto idx = sample_indices(X.size(), n_batch_size, do_boostrap, gen);

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
                for (unsigned int j = 0; j < n_batch_size; ++j) {
                    //auto lidx = leaf_idx[i][j];
                    auto lidx = _trees[i].leaf_index(X[idx[j]]);

                    for (unsigned int k = 0; k < n_classes; ++k) {
                        output[j][k] += _weights[i] * _trees[i]._leafs[lidx + k];
                    }
                }
            }

            if constexpr(tree_opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                
                #pragma omp parallel for
                for (unsigned int i = 0; i < _weights.size(); ++i) {
                    std::vector<internal_t> loss_deriv(n_classes, 0);
                    std::vector<internal_t> grad(_trees[i]._leafs.size(), 0);

                    // Compute gradient for current tree
                    for (unsigned int k = 0; k < n_batch_size; ++k) {
                        
                        // No need to reset loss_deriv because it will be copied anyway
                        loss.deriv(&output[k][0], &loss_deriv[0], Y[idx[k]], n_classes);

                        //auto lidx = leaf_idx[i][k];
                        auto lidx = _trees[i].leaf_index(X[k]);
                        for (unsigned int j = 0; j < n_classes; ++j) {
                            grad[lidx+j] += loss_deriv[j] * _weights[i] * 1.0 / n_batch_size * 1.0 / n_classes;
                        }
                    }
                    // Update current tree
                    _trees[i].optimizer.step(_trees[i]._leafs, grad);
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
                    for (unsigned int j = 0; j < n_batch_size; ++j) {
                        loss.deriv(&output[j][0], &loss_deriv[0], Y[idx[j]], n_classes);

                        //auto lidx = leaf_idx[i][j];
                        auto lidx = _trees[i].leaf_index(X[j]);
                        for (unsigned int k = 0; k < n_classes; ++k) {
                            dir += _trees[i]._leafs[lidx + k] * loss_deriv[k];
                        }
                    }
                    grad[i] = dir / (n_batch_size * n_classes);
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
                        unsigned int tidx = nonzero_idx[i];
                        _weights[tidx] = nonzero_w[i];
                    }
                }
            }
        }
    }

    void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int burnin_steps) {
        _weights.push_back(0.0);
        _trees.push_back(DecisionTree<tree_init, tree_opt>(n_classes,max_depth, max_features, seed++, step_size));
        _trees.back().fit(X,Y);
        
        update_trees(X, Y, burnin_steps, std::nullopt, std::nullopt, seed);
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

    void load(std::vector<std::vector<internal_t>> & new_nodes, std::vector<std::vector<internal_t>> & new_leafs, std::vector<internal_t> & new_weights) {
        
        _trees.clear();
        _weights = new_weights;

        for (unsigned int i = 0; i < new_weights.size(); ++i) {
            _trees.push_back(DecisionTree<tree_init, tree_opt>(n_classes, max_depth, max_features, seed+i, step_size));
            _trees.back().load(new_nodes[i], new_leafs[i]);
        }
    }

    std::tuple<std::vector<std::vector<internal_t>>, std::vector<std::vector<internal_t>>, std::vector<internal_t>> store() const {
        std::vector<std::vector<internal_t>> all_leafs(_trees.size());
        std::vector<std::vector<internal_t>> all_nodes(_trees.size());

        for (unsigned int i = 0;i < _trees.size(); ++i) {
            auto tmp = _trees[i].store();
            all_nodes[i] = std::get<0>(tmp);
            all_leafs[i] = std::get<1>(tmp);
        }

        return std::make_tuple(all_nodes, all_leafs, _weights); 
    }

    unsigned int num_nodes() const {
        unsigned int n_nodes = 0;
        for (auto const & t : _trees) {
            n_nodes += t.num_nodes();
        }
        return n_nodes;
    }

    unsigned int num_bytes() const {
        unsigned int tree_size = 0;
        for (auto const & t : _trees) {
            tree_size += t.num_bytes();
        }

        return tree_size + sizeof(*this) + optimizer.num_bytes();
    }

    unsigned int num_trees() const {
        return _trees.size();
    }

    // std::vector<internal_t> & weights() {
    //     return _weights;
    // }

    // std::vector<internal_t> & trees() {
    //     return _trees;
    // }

    // std::vector<Tree*> trees() {
    //     std::vector<Tree*> tree_ptrs(_trees.size());
    //     for (unsigned int i = 0;i < _trees.size(); ++i) {
    //         tree_ptrs[i] = &_trees[i];
    //     }
    //     return tree_ptrs;
    // }

};

#endif