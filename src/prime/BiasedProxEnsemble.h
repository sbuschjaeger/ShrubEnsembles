#ifndef BIASED_PROX_ENSEMBLE_H
#define BIASED_PROX_ENSEMBLE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>

#include "Datatypes.h"
#include "Tree.h"
#include "Losses.h"
#include "EnsembleRegularizer.h"
#include "TreeRegularizer.h"

enum class INIT_MODE {CONSTANT, AVERAGE, MAX};

void scale(std::vector<std::vector<data_t>> &X, data_t s) {
    for (unsigned int j = 0; j < X.size(); ++j) {
        for (unsigned int k = 0; k < X[j].size(); ++k) {
            X[j][k] *= s;
        }
    }
}

data_t mean_all_dim(std::vector<std::vector<data_t>> &X) {
    unsigned int n_first = X.size();
    unsigned int n_second = X[0].size();
    data_t mean = 0;

    for (unsigned int j = 0; j < n_first; ++j) {
        for (unsigned int k = 0; k < n_second; ++k) {
            mean += X[j][k];
        }
    }

    return mean / (n_first * n_second);
}

std::vector<std::vector<data_t>> weighted_sum_first_dim(std::vector<std::vector<std::vector<data_t>>> &X, std::vector<data_t> const &scale) {
    //unsigned int n_first = X.size();
    unsigned int n_second = X[0].size();
    unsigned int n_third = X[0][0].size();

    std::vector<std::vector<data_t>> XMean(n_second, std::vector<data_t> (n_third, 0));

    for (unsigned int i = 0; i < X.size(); ++i) {
        for (unsigned int j = 0; j < n_second; ++j) {
            for (unsigned int k = 0; k < n_third; ++k) {
                XMean[j][k] += X[i][j][k] * scale[i];
            }
        }
    }

    return XMean;
}

/**
 * @brief  The main reason why this interface exists, is because it makes class instansiation a little easier for the Pythonbindings. See Python.cpp for details.
 * @note   
 * @retval None
 */
template <typename pred_t>
class BiasedProxedEnsembleInterface {
public:
    virtual data_t next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) = 0;

    virtual std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) = 0;
    
    virtual std::vector<data_t> weights() const = 0;

    virtual unsigned int num_trees() const = 0;

    virtual ~BiasedProxedEnsembleInterface() { }
};

//TODO python interface
//TODO comments

template <TREE_INIT tree_init, TREE_NEXT tree_next, typename pred_t>
class BiasedProxEnsemble : public BiasedProxedEnsembleInterface<pred_t> {

private:
    std::vector< Tree<tree_init, tree_next, pred_t> > _trees;
    std::vector<data_t> _weights;

    unsigned int const n_classes;
    unsigned int const max_depth;
    unsigned long seed;
    bool const normalize_weights;

    data_t step_size;
    std::function< std::vector<std::vector<data_t>>(std::vector<std::vector<data_t>> const &, std::vector<unsigned int> const &) > loss;
    std::function< std::vector<std::vector<data_t>>(std::vector<std::vector<data_t>> const &, std::vector<unsigned int> const &) > loss_deriv;

    INIT_MODE const init_mode;
    data_t const init_weight;
    std::vector<bool> const is_nominal;
    
    std::function< std::vector<data_t>(std::vector<data_t> const &, data_t scale) > ensemble_regularizer;
    data_t const l_ensemble_reg;
    
    std::function< data_t(Tree<tree_init, tree_next, pred_t> const &) > tree_regularizer;
    data_t const l_tree_reg;

public:

    BiasedProxEnsemble(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        bool normalize_weights = true,
        LOSS loss = LOSS::MSE,
        data_t step_size = 1e-5,
        INIT_MODE init_mode = INIT_MODE::CONSTANT,
        data_t init_weight = 0.0,
        std::vector<bool> const & is_nominal = {},
        ENSEMBLE_REGULARIZER::TYPE ensemble_regularizer = ENSEMBLE_REGULARIZER::TYPE::NO,
        data_t l_ensemble_reg = 0.0,
        TREE_REGULARIZER::TYPE tree_regularizer = TREE_REGULARIZER::TYPE::NO,
        data_t l_tree_reg = 0.0
    ) : 
        n_classes(n_classes), 
        max_depth(max_depth), 
        seed(seed), 
        normalize_weights(normalize_weights), 
        step_size(step_size), 
        loss(loss_from_enum(loss)), 
        loss_deriv(loss_deriv_from_enum(loss)), 
        init_mode(init_mode), 
        init_weight(init_weight), 
        is_nominal(is_nominal), 
        ensemble_regularizer(ENSEMBLE_REGULARIZER::from_enum(ensemble_regularizer)), 
        l_ensemble_reg(l_ensemble_reg),
        tree_regularizer(TREE_REGULARIZER::from_enum<tree_init, tree_next, pred_t>(tree_regularizer)),
        l_tree_reg(l_tree_reg) 
    {}

    BiasedProxEnsemble(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        bool normalize_weights = true,
        std::function< std::vector<std::vector<data_t>>(std::vector<std::vector<data_t>> const &, std::vector<unsigned int> const &) > loss = mse,
        std::function< std::vector<std::vector<data_t>>(std::vector<std::vector<data_t>> const &, std::vector<unsigned int> const &) > loss_deriv = mse_deriv,
        data_t step_size = 1e-5,
        INIT_MODE init_mode = INIT_MODE::CONSTANT,
        data_t init_weight = 0.0,
        std::vector<bool> const & is_nominal = {},
        std::function< std::vector<data_t>(std::vector<data_t> const &, data_t scale) > ensemble_regularizer = ENSEMBLE_REGULARIZER::no_reg,
        data_t l_ensemble_reg = 0.0,
        std::function< data_t(Tree<tree_init, tree_next, pred_t>) const &> tree_regularizer = TREE_REGULARIZER::tree_no_reg,
        data_t l_tree_reg = 0.0
    ) : 
        n_classes(n_classes), 
        max_depth(max_depth), 
        seed(seed), 
        normalize_weights(normalize_weights), 
        step_size(step_size), 
        loss(loss), 
        loss_deriv(loss_deriv), 
        init_mode(init_mode), 
        init_weight(init_weight), 
        is_nominal(is_nominal), 
        ensemble_regularizer(ensemble_regularizer), 
        l_ensemble_reg(l_ensemble_reg),
        tree_regularizer(tree_regularizer),
        l_tree_reg(l_tree_reg) 
    {}

    // BiasedProxEnsemble(
    //     unsigned int max_depth,
    //     unsigned int n_classes, 
    //     unsigned long seed, 
    //     bool normalize_weights,
    //     INIT_MODE init_mode,
    //     data_t step_size,
    //     data_t l_ensemble_reg,
    //     data_t l_tree_reg,
    //     data_t init_weight,
    //     std::vector<bool> const &is_nominal,
    //     std::function< std::vector<std::vector<data_t>>(std::vector<std::vector<data_t>> const &, std::vector<unsigned int> const &) > loss,
    //     std::function< std::vector<std::vector<data_t>>(std::vector<std::vector<data_t>> const &, std::vector<unsigned int> const &) > loss_deriv,
    //     std::function< std::vector<data_t>(std::vector<data_t> const &, data_t scale) > ensemble_regularizer,
    //     std::function< data_t(Tree<tree_init, tree_next, pred_t>) const &> tree_regularizer
    //     // std::function< xt::xarray<data_t>(xt::xarray<data_t> &)> reg,
    //     // std::function< xt::xarray<data_t>(xt::xarray<data_t> &, data_t, data_t)> prox
    // ) : max_depth(max_depth), n_classes(n_classes), seed(seed), normalize_weights(normalize_weights), init_mode(init_mode), step_size(step_size), l_ensemble_reg(l_ensemble_reg), l_tree_reg(l_tree_reg), init_weight(init_weight), is_nominal(is_nominal), loss(loss), loss_deriv(loss_deriv), ensemble_regularizer(ensemble_regularizer), tree_regularizer(tree_regularizer) /*, reg(reg), prox(prox) */ {}

    data_t next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        // Create new tree
        if (_weights.size() == 0 || init_mode == INIT_MODE::CONSTANT) {
            _weights.push_back(init_weight);
        } else {
            if (init_mode == INIT_MODE::AVERAGE) {
                data_t average = accumulate( _weights.begin(), _weights.end(), 0.0) / _weights.size(); 
                _weights.push_back(average);
            } else {
                data_t max = *std::max_element(_weights.begin(), _weights.end());
                _weights.push_back(max);
            }
        }
        
        _trees.push_back(Tree<tree_init, tree_next, pred_t>(max_depth, n_classes, seed++, X, Y, is_nominal));

        std::vector<std::vector<std::vector<data_t>>> all_proba(_trees.size());
        for (unsigned int i = 0; i < _trees.size(); ++i) {
            all_proba[i] = _trees[i].predict_proba(X);
        }

        std::vector<std::vector<data_t>> output = weighted_sum_first_dim(all_proba, _weights);
        std::vector<std::vector<data_t>> losses = loss(output, Y);
        std::vector<std::vector<data_t>> losses_deriv = loss_deriv(output, Y);

        data_t reg_loss = mean_all_dim(losses); //+ lambda * reg(w_tensor);

        for (unsigned int i = 0; i < _trees.size(); ++i) {
            _trees[i].next(X, Y, losses_deriv, _weights[i] * step_size);
        }

        std::vector<data_t> directions(_weights.size(), 0);
        for (unsigned int i = 0; i < all_proba.size(); ++i) {
            for (unsigned int j = 0; j < all_proba[i].size(); ++j) {
                for (unsigned int k = 0; k < all_proba[i][j].size(); ++k) {
                    directions[i] += all_proba[i][j][k] * losses_deriv[j][k];
                }
            }
            directions[i] /= (X.size() * n_classes);

            if (l_tree_reg > 0) {
                directions[i] += l_tree_reg * tree_regularizer(_trees[i]);
            }
        }

        // tmp_w = self.estimator_weights_ - self.step_size*directions - self.step_size*node_deriv
        for (unsigned int i = 0; i < _weights.size(); ++i) {
            _weights[i] = _weights[i] - step_size * directions[i];
        }
        _weights = ensemble_regularizer(_weights, l_ensemble_reg);

        if (normalize_weights && _weights.size() > 0) {
            std::vector<data_t> nonzero_w(_weights.size());
            std::vector<unsigned int> nonzero_idx(_weights.size());
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

        return reg_loss;
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        std::vector<std::vector<data_t>> output(X.size());
        if (_trees.size() == 0) {
            for (unsigned int i = 0; i < X.size(); ++i) {
                std::vector<data_t> tmp(n_classes);
                std::fill(tmp.begin(), tmp.end(), 1.0/n_classes);
                output[i] = tmp;
            }
        } else {
            std::vector<std::vector<std::vector<data_t>>> all_proba(_trees.size());
            for (unsigned int i = 0; i < _trees.size(); ++i) {
                all_proba[i] = _trees[i].predict_proba(X);
            }
            output = weighted_sum_first_dim(all_proba, _weights);
        }
        return output;
    }

    unsigned int num_trees() const {
        return _trees.size();
    }

    std::vector<data_t> weights() const {
        return _weights;
    }
};

#endif