#ifndef ONLINE_SHRUB_ENSEMBLE_H
#define ONLINE_SHRUB_ENSEMBLE_H

#include <vector>
#include <math.h>
#include <random>
#include <algorithm>

#include "ShrubEnsemble.h"
#include "Optimizer.h"

/**
 * @brief  The main reason why this interface exists, is because it makes class instansiation a little easier for the Pythonbindings. See Python.cpp for details.
 * @note   
 * @retval None
 */
class OnlineShrubEnsembleInterface {
public:
    virtual void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) = 0;
    
    virtual void init_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size) = 0;

    virtual std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) = 0;
    
    virtual std::vector<internal_t> weights() const = 0;

    virtual unsigned int num_trees() const = 0;

    virtual unsigned int num_bytes() const = 0;
    
    virtual unsigned int num_nodes() const = 0;

    virtual ~OnlineShrubEnsembleInterface() { }
};

template <OPTIMIZER::OPTIMIZER_TYPE opt, TREE_INIT tree_init>
class OnlineShrubEnsemble : public OnlineShrubEnsembleInterface, private ShrubEnsemble<opt, OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_init> {
    
private:

public:

    OnlineShrubEnsemble(
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
    ) : ShrubEnsemble<opt, OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_init>(
            n_classes,
            max_depth,
            seed,
            normalize_weights,
            burnin_steps,
            max_features,
            loss,
            step_size,
            ensemble_regularizer,
            l_ensemble_reg,
            tree_regularizer,
            l_tree_reg) {}

    void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        ShrubEnsemble<opt, OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_init>::next(X,Y);
    }
    
    void init_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size) {
        ShrubEnsemble<opt, OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_init>::init_trees(X,Y,n_trees,bootstrap,batch_size);
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        return ShrubEnsemble<opt, OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_init>::predict_proba(X);
    }
    
    std::vector<internal_t> weights() const {
        return ShrubEnsemble<opt, OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_init>::weights();
    }

    unsigned int num_trees()const {
        return ShrubEnsemble<opt, OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_init>::num_trees();
    }

    unsigned int num_bytes()const {
        return ShrubEnsemble<opt, OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_init>::num_bytes();
    }
    
    unsigned int num_nodes() const {
        return ShrubEnsemble<opt, OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_init>::num_nodes();
    }
};

#endif