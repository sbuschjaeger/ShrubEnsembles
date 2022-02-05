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

void scale(std::vector<std::vector<data_t>> &X, data_t s) {
    for (unsigned int j = 0; j < X.size(); ++j) {
        for (unsigned int k = 0; k < X[j].size(); ++k) {
            X[j][k] *= s;
        }
    }
}

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

auto sample_data(std::vector<std::vector<data_t>>const &X, std::vector<unsigned int>const &Y, unsigned int batch_size, bool bootstrap, long seed = 1234) {
    if (batch_size >= X.size() || batch_size == 0) {
        batch_size = X.size();
    }

    std::vector<std::vector<data_t>> bX(batch_size);
    std::vector<unsigned int> bY(batch_size);

    if (bootstrap) {
        auto gen = std::default_random_engine(seed);
        std::uniform_int_distribution<> dist(0, X.size()-1); 
        for (unsigned int i = 0; i < batch_size; ++i) {
            auto idx = dist(gen);
            bX[i] = X[idx];
            bY[i] = Y[idx];
        }
    } else {
        std::vector<unsigned int> idx(X.size());
        std::iota(std::begin(idx), std::end(idx), 0);
        std::shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));
        for (unsigned int i = 0; i < batch_size; ++i) {
            bX[i] = X[idx[i]];
            bY[i] = Y[idx[i]];
        }
    }

    return std::make_tuple(bX, bY);
}

// /**
//  * @brief  The main reason why this interface exists, is because it makes class instansiation a little easier for the Pythonbindings. See Python.cpp for details.
//  * @note   
//  * @retval None
//  */
// class ShrubEnsembleInterface {
// public:
//     virtual void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) = 0;
    
//     virtual void init_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size) = 0;

//     virtual void next_distributed(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_parallel, bool bootstrap, unsigned int batch_size) = 0;

//     virtual void fit_distributed(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size, unsigned int n_rounds) = 0;

//     virtual std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) = 0;
    
//     virtual std::vector<internal_t> weights() const = 0;

//     virtual unsigned int num_trees() const = 0;

//     virtual unsigned int num_bytes() const = 0;
    
//     virtual unsigned int num_nodes() const = 0;

//     virtual ~ShrubEnsembleInterface() { }
// };

template <OPTIMIZER::OPTIMIZER_TYPE opt, OPTIMIZER::OPTIMIZER_TYPE tree_opt, TREE_INIT tree_init>
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

    std::function< std::vector<std::vector<internal_t>>(std::vector<std::vector<internal_t>> const &, std::vector<unsigned int> const &) > loss;
    std::function< std::vector<std::vector<internal_t>>(std::vector<std::vector<internal_t>> const &, std::vector<unsigned int> const &) > loss_deriv;

    std::function< std::vector<internal_t>(std::vector<internal_t> const &, data_t scale) > ensemble_regularizer;
    data_t const l_ensemble_reg;
    
    std::function< internal_t(Tree<tree_init, tree_opt> const &) > tree_regularizer;
    internal_t const l_tree_reg;

public:

    ShrubEnsemble(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        bool normalize_weights = true,
        unsigned int burnin_steps = 0,
        unsigned int max_features = 0,
        LOSS::TYPE loss = LOSS::TYPE::MSE,
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
        loss(LOSS::from_enum(loss)), 
        loss_deriv(LOSS::deriv_from_enum(loss)), 
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

    void prune() {
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
            optimizer.reset();
        }
    }

    void init_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool boostrap, unsigned int batch_size) {
        for (unsigned int i = 0; i < n_trees; ++i) {
            _trees.push_back(Tree<tree_init, tree_opt>(n_classes, max_depth, max_features, seed, step_size));
            seed++;
            _weights.push_back(1.0 / n_trees);    
        }

        #pragma omp parallel for
        for (unsigned int i = 0; i < n_trees; ++i){
            auto s = sample_data(X, Y, batch_size, boostrap, seed+i);
            _trees[i].fit(std::get<0>(s), std::get<1>(s));
        }
        seed++;
    }

    void next_distributed(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_parallel, bool boostrap, unsigned int batch_size) {
        std::vector<ShrubEnsemble<opt, tree_opt, tree_init>> ses(_trees.size(), *this);

        #pragma omp parallel for
        for (unsigned int k = 0; k < n_parallel; ++k){
            auto s = sample_data(X, Y, batch_size, boostrap, seed+k);
            ses[k].update_trees(std::get<0>(s), std::get<1>(s));
        }
        seed++;

        #pragma omp parallel for 
        for (unsigned int j = 0; j < _trees.size(); ++j) {
            for (unsigned int k = 0; k < ses.size(); ++k) {
                auto & cur = ses[k];
                if ( k == 0) {
                    _weights[j] = 1.0 / ses.size() * cur._weights[j];
                    _trees[j].leafs = cur._trees[j].leafs;
                } else {
                    _weights[j] += 1.0 / ses.size() * cur._weights[j];

                    for (unsigned int l = 0; l < cur._trees[j].leafs.size(); ++l) {
                        _trees[j].leafs[l] += 1.0 / ses.size() * cur._trees[j].leafs[l];
                    }
                }
            }
        }
    }

    void fit_distributed(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size, unsigned int n_rounds) {
        init_trees(X, Y, n_trees, bootstrap, batch_size);
        
        for (unsigned int i = 0; i < n_rounds; ++i) {
            next_distributed(X,Y,n_trees,bootstrap,batch_size);

            // auto output = this->predict_proba(X);
            // std::vector<std::vector<data_t>> losses = loss(output, Y);

            prune();
        }
    }

    void update_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        std::vector<std::vector<std::vector<internal_t>>> all_proba(_trees.size());
        if (_trees.size() > 0) {
            for (unsigned int i = 0; i < _trees.size(); ++i) {
                all_proba[i] = _trees[i].predict_proba(X);
            }
        } 

        for (unsigned int s = 0; s < burnin_steps + 1; ++s) {
            std::vector<std::vector<internal_t>> output;
            output = weighted_sum_first_dim(all_proba, _weights);
            // if (_trees.size() > 0) {
            //     output = weighted_sum_first_dim(all_proba, _weights);
            // } 

            //std::vector<std::vector<data_t>> losses = loss(output, Y);
            std::vector<std::vector<internal_t>> losses_deriv = loss_deriv(output, Y);

            //data_t reg_loss = mean_all_dim(losses); //+ lambda * reg(w_tensor);
            if constexpr(tree_opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                #pragma omp parallel for
                for (unsigned int i = 0; i < _weights.size(); ++i) {
                    // If we are not in the last round of burn-in and i is the latest tree, then dont update the tree 
                    //if (j < burnin_steps && i == _weights.size() - 1) continue;

                    // The update for a single tree is cur_leaf = cur_leaf - step_size * tree_grad 
                    // where tree_grad = _weights[i] * loss_deriv
                    // Thus, _weights[i] should be be part of losses_deriv. But for simplictiy, we will 
                    // simply use _weights[i] * step_size as "step size" here
                
                    std::vector<internal_t> grad(_trees[i].leafs.size(), 0);
                    for (unsigned int k = 0; k < X.size(); ++k) {
                        auto idx = _trees[i].leaf_index(X[k]);
                        for (unsigned int j = 0; j < n_classes; ++j) {
                            grad[idx+j] += losses_deriv[k][j] * _weights[i];
                        }
                        _trees[i].optimizer.step(_trees[i].leafs, grad);
                    }
                }
            }

            if constexpr(opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                std::vector<internal_t> grad(_weights.size(), 0);

                #pragma omp parallel for
                for (unsigned int i = 0; i < all_proba.size(); ++i) {
                    internal_t dir = 0;
                    for (unsigned int j = 0; j < all_proba[i].size(); ++j) {
                        for (unsigned int k = 0; k < all_proba[i][j].size(); ++k) {
                            dir += all_proba[i][j][k] * losses_deriv[j][k];
                        }
                    }
                    dir /= (X.size() * n_classes);

                    if (l_tree_reg > 0) {
                        for (unsigned int i = 0; i < _trees.size(); ++i) {
                            dir += l_tree_reg * tree_regularizer(_trees[i]);
                        }
                    }
                    grad[i] = dir;
                }

                // std::vector<internal_t> grad(_weights.size(), dir);
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

        prune();
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        std::vector<std::vector<data_t>> output; 
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