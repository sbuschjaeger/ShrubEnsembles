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
#include "EnsembleRegularizer.h"
#include "TreeRegularizer.h"

#include "Utils.h"

/**
 * @brief  
 * @note   
 * @retval None
 */

template <LOSS::TYPE loss_type, OPTIMIZER::OPTIMIZER_TYPE opt, OPTIMIZER::OPTIMIZER_TYPE tree_opt>
//template <OPTIMIZER::OPTIMIZER_TYPE tree_opt, DT::TREE_INIT tree_init>
class TreeEnsemble {

protected:
    // DecisionTree<tree_init, tree_opt>
    std::vector<Tree<tree_opt> *> _trees;
    std::vector<internal_t> _weights;
    Tree<tree_opt> * prototype;

    unsigned int const n_classes;
    unsigned long seed;

    bool const normalize_weights;

    OPTIMIZER::Optimizer<opt> optimizer;

    LOSS::Loss<loss_type> loss;

    std::function< std::vector<internal_t>(std::vector<internal_t> const &, data_t scale) > ensemble_regularizer;
    internal_t const l_ensemble_reg;
    
    std::function< internal_t(Tree<tree_opt> const &) > tree_regularizer;
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
        Tree<tree_opt> * prototype, 
        unsigned long seed = 12345,
        bool normalize_weights = true,
        internal_t step_size = 1e-2,
        ENSEMBLE_REGULARIZER::TYPE ensemble_regularizer = ENSEMBLE_REGULARIZER::TYPE::NO,
        internal_t l_ensemble_reg = 0.0,
        TREE_REGULARIZER::TYPE tree_regularizer = TREE_REGULARIZER::TYPE::NO,
        internal_t l_tree_reg = 0.0
    ) : 
        n_classes(n_classes), 
        prototype(prototype),
        seed(seed), 
        normalize_weights(normalize_weights), 
        ensemble_regularizer(ENSEMBLE_REGULARIZER::from_enum(ensemble_regularizer)), 
        l_ensemble_reg(l_ensemble_reg),
        tree_regularizer(TREE_REGULARIZER::from_enum<tree_opt>(tree_regularizer)),
        l_tree_reg(l_tree_reg)
    {
        if (prototype == nullptr) {
            throw std::runtime_error("Received a prototype that was a nullptr and hence cannot clone any objects!");
        }
    }

    ~TreeEnsemble() {
        for (unsigned int i = 0; i < _trees.size(); ++i) {
            delete _trees[i];
        }
        delete prototype;
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

    void init(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, unsigned int n_trees, bool boostrap, unsigned int batch_size) {
        
        // Create the tree objects and initialize their weight. 
        // We do this in a single thread so that we can perform the training without any
        // synchroization
        for (unsigned int i = 0; i < n_trees; ++i) {
            _trees.push_back(prototype->clone(seed));
            _weights.push_back(1.0 / n_trees);    
        }
        // Make sure to advance the random seed "properly"
        seed += n_trees;

        // Do the training in parallel
        #pragma omp parallel for
        for (unsigned int i = 0; i < n_trees; ++i){
            auto idx = sample_indices(X.rows, batch_size, boostrap, seed + i);
            // TODO CALL DIFFERENT FIT METHOD HERE
            _trees[i]->fit(X,Y,idx);

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
            //if (_trees[0].optimizer().step_size > 0) {
            if constexpr(tree_opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                
                #pragma omp parallel for
                for (unsigned int i = 0; i < _weights.size(); ++i) {
                    matrix1d<internal_t> loss_deriv(n_classes);
                    std::fill(loss_deriv.begin(), loss_deriv.end(), 0);

                    matrix1d<internal_t> grad(_trees[i]->leaves().size());
                    std::fill(grad.begin(), grad.end(), 0);

                    // Compute gradient for current tree
                    for (unsigned int k = 0; k < n_batch_size; ++k) {
                        
                        // No need to reset loss_deriv because it will be copied anyway
                        loss.deriv(output(k), loss_deriv, Y(idx(k)));

                        //auto lidx = leaf_idx[i][k];
                        auto lidx = _trees[i]->leaf_index(X(idx(k)));
                        for (unsigned int j = 0; j < n_classes; ++j) {
                            // grad[lidx+j] += loss_deriv[j] * _weights[i] * 1.0 / n_batch_size * 1.0 / n_classes;
                            grad(lidx+j) += loss_deriv(j) * _weights[i] * 1.0 / n_batch_size * 1.0 / n_classes;
                        }
                    }
                    // Update current tree
                    matrix1d<internal_t> tleafs(_trees[i]->leaves().size(), &_trees[i]->leaves()[0], false);
                    _trees[i]->optimizer().step(tleafs, grad);
                    // _trees[i]->optimizer.step(_trees[i]->leaves(), grad);
                }
            }
        
            if constexpr(opt != OPTIMIZER::OPTIMIZER_TYPE::NONE) {
                // Compute gradient for the weights
                matrix1d<internal_t> grad(_weights.size());
                std::fill(grad.begin(), grad.end(), 0); 

                #pragma omp parallel for
                for (unsigned int i = 0; i < _trees.size(); ++i) {
                    matrix1d<internal_t> loss_deriv(n_classes);
                    std::fill(loss_deriv.begin(), loss_deriv.end(), 0); 

                    internal_t dir = 0;

                    // Compute tree regularization if necessary
                    if (l_tree_reg > 0) {
                        dir += l_tree_reg * tree_regularizer(_trees[i]);
                    }

                    // Compute gradient for tree i
                    for (unsigned int j = 0; j < n_batch_size; ++j) {
                        // auto joutput = output(j);
                        loss.deriv(output(j), loss_deriv, Y(idx(j)));

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
                optimizer.step(tweights, grad);
                // optimizer.step(_weights, grad);
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

    matrix2d<internal_t> predict_proba(matrix2d<data_t> const &X) {
        if (_trees.size() == 0) {
            matrix2d<internal_t> output(X.rows, n_classes); 
            std::fill(output.begin(), output.end(), 1.0/n_classes);
            return output;
        } else {
            matrix3d<internal_t> all_proba(_trees.size(), X.rows, n_classes);
            for (unsigned int i = 0; i < _trees.size(); ++i) {
                auto tmp = all_proba(i);
                _trees[i]->predict_proba(X, tmp);
            }
            matrix1d<internal_t> tweights(_weights.size(), &_weights[0], false);
            matrix2d<internal_t> output = weighted_sum_first_dim(all_proba, tweights);
            return output;
        }
    }

    void load(std::vector<matrix1d<internal_t>> const & new_nodes, std::vector<matrix1d<internal_t>> const & new_leafs, std::vector<internal_t> const & new_weights) {
        _trees.clear();
        _weights = std::vector<internal_t>(new_weights);

        for (unsigned int i = 0; i < new_weights.size(); ++i) {
            _trees.push_back(prototype->clone(seed+i));
            //_trees.push_back(DecisionTree<tree_init, tree_opt>(n_classes, max_depth, max_features, seed+i, step_size));
            _trees.back().load(new_nodes[i], new_leafs[i]);
        }
    }

    std::tuple<std::vector<matrix1d<internal_t>>, std::vector<matrix1d<internal_t>>, std::vector<internal_t>> store() const {
        std::vector<matrix1d<internal_t>> all_leafs(_trees.size());
        std::vector<matrix1d<internal_t>> all_nodes(_trees.size());

        for (unsigned int i = 0;i < _trees.size(); ++i) {
            auto tmp = _trees[i]->store();
            all_nodes[i] = std::move(std::get<0>(tmp));
            all_leafs[i] = std::move(std::get<1>(tmp));
        }

        return std::make_tuple<std::vector<matrix1d<internal_t>>, std::vector<matrix1d<internal_t>>, std::vector<internal_t>>(std::move(all_nodes), std::move(all_leafs), std::vector(_weights)); 
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
};