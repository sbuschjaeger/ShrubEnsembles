#pragma once

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>
#include <optional>
#include <iostream>

#include "Datatypes.h"
#include "DecisionTree.h"
#include "Losses.h"


#include "Utils.h"

template <typename data_t>
internal_t nodes_reg(Tree<data_t> const & tree) {
    return tree.num_nodes();
}

std::vector<internal_t> L0_reg(std::vector<internal_t> const &w, internal_t scale) {
    std::vector<internal_t> tmp_w(w.size());
    internal_t tmp = std::sqrt(2 * scale);
    for (unsigned int i = 0; i < w.size(); ++i) {
        if (std::abs(w[i]) < tmp) {
            tmp_w[i] = 0;
        } else {
            tmp_w[i] = w[i];
        }
    }

    return tmp_w;
}

std::vector<internal_t> L1_reg(std::vector<internal_t> const &w, internal_t scale) {
    std::vector<internal_t> tmp_w(w.size());
    for (unsigned int i = 0; i < w.size(); ++i) {
        internal_t sign = w[i] > 0 ? 1 : -1;
        tmp_w[i] = sign * std::max(0.0, std::abs(w[i])  - scale);
    }

    return tmp_w;
}

std::vector<internal_t> L2_reg(std::vector<internal_t> const &w, internal_t scale) {
    std::vector<internal_t> tmp_w(w.size());
    for (unsigned int i = 0; i < w.size(); ++i) {
        tmp_w[i] = 2*w[i];
    }

    return tmp_w;
}

// https://stackoverflow.com/questions/14902876/indices-of-the-k-largest-elements-in-an-unsorted-length-n-array
std::vector<unsigned int> top_k(std::vector<internal_t> const &a, unsigned int K) {
    std::vector<unsigned int> top_idx;
    std::priority_queue< std::pair<internal_t, unsigned int>, std::vector< std::pair<internal_t, unsigned int> >, std::greater <std::pair<internal_t, unsigned int> > > q;
  
    for (unsigned int i = 0; i < a.size(); ++i) {
        if (q.size() < K) {
            q.push(std::pair<internal_t, unsigned int>(a[i], i));
        } else if (q.top().first < a[i]) {
            q.pop();
            q.push(std::pair<internal_t, unsigned int>(a[i], i));
        }
    }

    while (!q.empty()) {
        top_idx.push_back(q.top().second);
        q.pop();
    }

    return top_idx;
}

std::vector<internal_t> hard_L0_reg(std::vector<internal_t> const &w, internal_t K) {
    std::vector<unsigned int> top_idx = top_k(w, K);
    std::vector<internal_t> tmp_w(w.size(), 0);

    for (auto i : top_idx) {
        tmp_w[i] = w[i];
    }

    return tmp_w;
}

std::vector<internal_t> to_prob_simplex(std::vector<internal_t> const &w) {
    if (w.size() == 0) {
        return w;
    }

    std::vector<internal_t> u(w);
    std::sort(u.begin(), u.end(), std::greater<int>());

    internal_t u_sum = 0; 
    internal_t l = 0;
    for (unsigned int i = 0; i < w.size(); ++i) {
        u_sum += u[i];
        internal_t tmp = 1.0 / (i + 1.0) * (1.0 - u_sum);
        if ((u[i] + tmp) > 0) {
            l = tmp;
        }
    }

    for (unsigned int i = 0; i < w.size(); ++i) {
        u[i] = std::max(w[i] + l, 0.0);
    }

    return u;
}

/**
 * @brief  
 * @note   
 * @retval None
 */

// template <typename data_tt>
// class MASE;

template <typename data_t, INIT tree_init>
//template <OPTIMIZER::OPTIMIZER_TYPE tree_opt, DT::TREE_INIT tree_init>
class TreeEnsemble {

template <typename data_tt, INIT tree_initt>
friend class MASE;

protected:
    std::vector<DecisionTree<data_t, tree_init>> _trees;
    unsigned int max_depth;
    unsigned int max_features;
    std::optional<std::function<internal_t(std::vector<unsigned int> const &, std::vector<unsigned int> const &)>> score;

    std::unique_ptr<Optimizer> tree_optimizer;
    std::unique_ptr<Optimizer> weight_optimizer;

    std::unique_ptr<Loss> loss;

    std::vector<internal_t> _weights;

    unsigned int const n_classes;
    unsigned long seed;

    bool const normalize_weights;

    std::optional<std::function< std::vector<internal_t>(std::vector<internal_t> const &, internal_t scale)>> ensemble_regularizer;
    internal_t const l_ensemble_reg;
    
    std::optional<std::function< internal_t(Tree<data_t> const &)>> tree_regularizer;
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

    TreeEnsemble(
        unsigned int n_classes, 
        unsigned int max_depth, 
        unsigned int max_features, 
        std::unique_ptr<Loss> loss,
        unsigned long seed = 12345,
        bool normalize_weights = true,
        std::unique_ptr<Optimizer> tree_optimizer = nullptr,
        std::unique_ptr<Optimizer> weight_optimizer = nullptr,
        std::optional<std::function< std::vector<internal_t>(std::vector<internal_t> const &, internal_t scale)>> ensemble_regulizer = std::nullopt,
        internal_t l_ensemble_reg = 0,
        std::optional<std::function< internal_t(Tree<data_t> const &)>> tree_regularizer = std::nullopt,
        internal_t l_tree_reg = 0
    ) : 
        n_classes(n_classes), 
        max_depth(max_depth),
        max_features(max_features),
        score(std::nullopt),
        loss(std::move(loss)),
        seed(seed), 
        normalize_weights(normalize_weights), 
        tree_optimizer(std::move(tree_optimizer)),
        weight_optimizer(std::move(weight_optimizer)),
        ensemble_regularizer(ensemble_regulizer), 
        l_ensemble_reg(l_ensemble_reg),
        tree_regularizer(tree_regularizer),
        l_tree_reg(l_tree_reg)
    {}

    TreeEnsemble(
        unsigned int n_classes, 
        unsigned int max_depth, 
        unsigned int max_features, 
        std::function< internal_t(std::vector<unsigned int> const &, std::vector<unsigned int> const &)> score,
        std::unique_ptr<Loss> loss,
        unsigned long seed = 12345,
        bool normalize_weights = true,
        std::unique_ptr<Optimizer> tree_optimizer = nullptr,
        std::unique_ptr<Optimizer> weight_optimizer = nullptr,
        std::optional<std::function< std::vector<internal_t>(std::vector<internal_t> const &, internal_t scale)>> ensemble_regulizer = std::nullopt,
        internal_t l_ensemble_reg = 0,
        std::optional<std::function< internal_t(Tree<data_t> const &)>> tree_regularizer = std::nullopt,
        internal_t l_tree_reg = 0
    ) : 
        n_classes(n_classes), 
        max_depth(max_depth),
        max_features(max_features),
        score(score),
        loss(std::move(loss)),
        seed(seed), 
        normalize_weights(normalize_weights), 
        tree_optimizer(std::move(tree_optimizer)),
        weight_optimizer(std::move(weight_optimizer)),
        ensemble_regularizer(ensemble_regulizer), 
        l_ensemble_reg(l_ensemble_reg),
        tree_regularizer(tree_regularizer),
        l_tree_reg(l_tree_reg)
    {}

    ~TreeEnsemble() {}

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
            weight_optimizer->reset();
        }
    }

    void init_trees(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, unsigned int n_trees, bool boostrap, unsigned int batch_size) {
        
        // Create the tree objects and initialize their weight. 
        // We do this in a single thread so that we can perform the training without any
        // synchroization
        for (unsigned int i = 0; i < n_trees; ++i) {
            if constexpr(tree_init == INIT::CUSTOM) {
                _trees.push_back(DecisionTree<data_t, tree_init>(n_classes, max_depth, max_features, seed, *score));
            } else {
                _trees.push_back(DecisionTree<data_t, tree_init>(n_classes, max_depth, max_features, seed));
            }
            _weights.push_back(1.0 / n_trees);    
            // if (loss) tree_optimizers.push_back(the_tree_optimizer->clone());
        }
        // Make sure to advance the random seed "properly"
        seed += n_trees;

        // Do the training in parallel
        #pragma omp parallel for
        for (unsigned int i = 0; i < n_trees; ++i){
            auto idx = sample_indices(X.rows, batch_size, boostrap, seed + i);
            // TODO somehow precompute the distance matrix if we are using distance trees?
            _trees[i].fit(X,Y,idx);

        }
        // Make sure to advance the random seed "properly"
        seed += n_trees;
    }

    void update_trees(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, unsigned int burnin_steps, std::optional<unsigned int> const batch_size = std::nullopt, std::optional<bool> bootstrap = std::nullopt,std::optional<unsigned long> seed = std::nullopt) {
        // The structure of the trees does not change with the optimization and hence we can 
        // pre-compute the leaf indices for each tree / sample and store them. This mitigates the
        // somewhat "costly" iteration of the trees in each round but gives direct access to the
        // leaf nodes

        unsigned int n_batch_size = X.rows;
        if (batch_size.has_value() && batch_size.value() > 0 && X.rows > batch_size) {
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

        // Store the current predictions in the output vector. 
        matrix2d<internal_t> output(n_batch_size, n_classes);
        
        for (unsigned int s = 0; s < burnin_steps + 1; ++s) {
            auto idx = sample_indices(X.rows, n_batch_size, do_boostrap, gen);

            // Reset the output vector because we "add into" it in the for loop below
            // Compute the predictions for each tree / sample with the pre-computed indices.
            // This can be done a bit more efficient if we would update the output vector after the gradient step
            // instead of recomputing the entire predictions from scratch. But this is more readable and 
            // maintainable
            std::fill(output.begin(), output.end(), static_cast<internal_t>(0));
            // for(auto & o : output) {
            //     std::fill(o.begin(), o.end(), static_cast<internal_t>(0));
            // }

            #pragma omp parallel for
            for (unsigned int i = 0; i < _trees.size(); ++i) {
                for (unsigned int j = 0; j < n_batch_size; ++j) {
                    //auto lidx = leaf_idx[i][j];
                    auto lidx = _trees[i]->leaf_index(X(idx(j)));

                    for (unsigned int k = 0; k < n_classes; ++k) {
                        output(j,k) += _weights[i] * _trees[i]->leaves()[lidx + k];
                    }
                }
            }

            // TODO This assumes a homogenous ensemble in which either all or no
            // tree is updated. 
            if (tree_optimizer != nullptr) {    
                #pragma omp parallel for
                for (unsigned int i = 0; i < _weights.size(); ++i) {
                    matrix1d<internal_t> loss_deriv(n_classes);
                    std::fill(loss_deriv.begin(), loss_deriv.end(), 0);

                    matrix1d<internal_t> grad(_trees[i]->leaves().size());
                    std::fill(grad.begin(), grad.end(), 0);

                    // Compute gradient for current tree
                    for (unsigned int k = 0; k < n_batch_size; ++k) {
                        
                        // No need to reset loss_deriv because it will be copied anyway
                        loss->deriv(output(k), loss_deriv, Y(idx(k)));

                        //auto lidx = leaf_idx[i][k];
                        auto lidx = _trees[i]->leaf_index(X(idx(k)));
                        for (unsigned int j = 0; j < n_classes; ++j) {
                            // grad[lidx+j] += loss_deriv[j] * _weights[i] * 1.0 / n_batch_size * 1.0 / n_classes;
                            grad(lidx+j) += loss_deriv(j) * _weights[i] * 1.0 / n_batch_size * 1.0 / n_classes;
                        }
                    }
                    // Update current tree
                    matrix1d<internal_t> tleafs(_trees[i]->leaves().size(), &_trees[i]->leaves()[0], false);
                    tree_optimizer->step(i, tleafs, grad);
                    
                    //_trees[i]->optimizer().step(tleafs, grad);
                    // _trees[i]->optimizer.step(_trees[i]->leaves(), grad);
                }
            }
        
            if (weight_optimizer) {
                // Compute gradient for the weights
                matrix1d<internal_t> grad(_weights.size());
                std::fill(grad.begin(), grad.end(), 0); 

                #pragma omp parallel for
                for (unsigned int i = 0; i < _trees.size(); ++i) {
                    matrix1d<internal_t> loss_deriv(n_classes);
                    std::fill(loss_deriv.begin(), loss_deriv.end(), 0); 

                    internal_t dir = 0;

                    // Compute tree regularization if necessary
                    if (l_tree_reg > 0 && tree_regularizer.has_value()) {
                        dir += l_tree_reg * (*tree_regularizer)(*_trees[i]);
                    }

                    // Compute gradient for tree i
                    for (unsigned int j = 0; j < n_batch_size; ++j) {
                        // auto joutput = output(j);
                        loss->deriv(output(j), loss_deriv, Y(idx(j)));

                        //auto lidx = leaf_idx[i][j];
                        auto lidx = _trees[i]->leaf_index(X(idx(j)));
                        for (unsigned int k = 0; k < n_classes; ++k) {
                            dir += _trees[i]->leaves()[lidx + k] * loss_deriv(k);
                        }
                    }
                    grad(i) = dir / (n_batch_size * n_classes);
                }

                // Perform SGD step for weights and apply prox operator afterwards
                matrix1d<internal_t> tweights(_weights.size(), &_weights[0], false);
                weight_optimizer->step(0, tweights, grad);
                // optimizer.step(_weights, grad);

                if (ensemble_regularizer.has_value()) _weights = ensemble_regularizer.value()(_weights, l_ensemble_reg);
            
                if (normalize_weights && _weights.size() > 0) {
                    std::vector<internal_t> nonzero_w;
                    std::vector<unsigned int> nonzero_idx;
                    for (unsigned int i = 0; i < _weights.size(); ++i) {
                        if (_weights[i] != 0) {
                            nonzero_w.push_back(_weights[i]);
                            nonzero_idx.push_back(i);
                        }
                    }
                    nonzero_w = to_prob_simplex(nonzero_w);
                    for (unsigned int i = 0; i < nonzero_idx.size(); ++i) {
                        unsigned int tidx = nonzero_idx[i];
                        _weights[tidx] = nonzero_w[i];
                    }
                }
            }
        }
    }

    matrix2d<internal_t> predict_proba(matrix2d<data_t> const &X) {
        if (_trees.size() == 0) {
            matrix2d<internal_t> output(X.rows, n_classes); 
            std::fill(output.begin(), output.end(), 1.0/n_classes);
            return output;
        } else {
            matrix3d<internal_t> all_proba(_trees.size(), X.rows, n_classes);
            for (unsigned int i = 0; i < _trees.size(); ++i) {
                auto tmp = all_proba(i);
                _trees[i].predict_proba(X, tmp);
            }
            matrix1d<internal_t> tweights(_weights.size(), &_weights[0], false);
            matrix2d<internal_t> output = weighted_sum_first_dim(all_proba, tweights);
            return output;
        }
    }

    void load(std::vector<matrix1d<internal_t>> const & new_nodes, std::vector<internal_t> const & new_weights) {
        _trees.clear();
        _weights = std::vector<internal_t>(new_weights);

        for (unsigned int i = 0; i < new_weights.size(); ++i) {
            if (score.has_value()) {
                _trees.push_back(DecisionTree<data_t, tree_init>(n_classes, max_depth, max_features, seed+i, score));
            } else {
                _trees.push_back(DecisionTree<data_t, tree_init>(n_classes, max_depth, max_features, seed+i));
            }
            //_trees.push_back(DecisionTree<tree_init, tree_opt>(n_classes, max_depth, max_features, seed+i, step_size));
            _trees.back()->load(new_nodes[i]);
        }
    }

    std::tuple<std::vector<matrix1d<internal_t>>, std::vector<internal_t>> store() const {
        std::vector<matrix1d<internal_t>> all_nodes(_trees.size());

        for (unsigned int i = 0;i < _trees.size(); ++i) {
            all_nodes[i] = std::move(_trees[i]->store());
        }

        return std::make_tuple<std::vector<matrix1d<internal_t>>, std::vector<internal_t>>(std::move(all_nodes), std::vector(_weights)); 
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

        if (tree_optimizer != nullptr) {
            tree_size += sizeof(std::unique_ptr<Optimizer>) * tree_optimizer->num_bytes();
        }

        if (weight_optimizer != nullptr) {
            tree_size += sizeof(std::unique_ptr<Optimizer>) * weight_optimizer->num_bytes();
        }

        return tree_size + sizeof(*this) + sizeof(internal_t) * _weights.size();
    }

    unsigned int num_trees() const {
        return _trees.size();
    }
};