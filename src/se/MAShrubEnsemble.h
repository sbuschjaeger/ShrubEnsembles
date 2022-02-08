#ifndef DISTRIBUTED_SHRUB_ENSEMBLE_H
#define DISTRIBUTED_SHRUB_ENSEMBLE_H

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
class MAShrubEnsembleInterface {
public:

    virtual void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) = 0;

    virtual void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) = 0;

    virtual void init(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) = 0;

    virtual std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) = 0;
    
    virtual std::vector<internal_t> weights() const = 0;

    virtual unsigned int num_trees() const = 0;

    virtual unsigned int num_bytes() const = 0;
    
    virtual unsigned int num_nodes() const = 0;

    virtual ~MAShrubEnsembleInterface() { }
};

template <OPTIMIZER::OPTIMIZER_TYPE tree_opt, TREE_INIT tree_init>
class MAShrubEnsemble : public MAShrubEnsembleInterface, private ShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_opt, tree_init> {
    
private:

    unsigned int n_trees;
    unsigned int n_rounds;
    unsigned int batch_size;
    bool bootstrap; 

public:

    MAShrubEnsemble(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        unsigned int burnin_steps = 0,
        unsigned int max_features = 0,
        LOSS::TYPE loss = LOSS::TYPE::MSE,
        internal_t step_size = 1e-2,
        unsigned int n_trees = 32, 
        unsigned int n_rounds = 5,
        unsigned int batch_size = 0,
        bool bootstrap = true
    ) : ShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_opt, tree_init>(
            n_classes,
            max_depth,
            seed,
            false,
            burnin_steps,
            max_features,
            loss,
            step_size), n_trees(n_trees), n_rounds(n_rounds), batch_size(batch_size), bootstrap(bootstrap)  {}

    void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        ShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_opt, tree_init>::fit_distributed(X,Y,n_trees,bootstrap,batch_size,n_rounds);
    }

    void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        ShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_opt, tree_init>::next_distributed(X,Y,n_trees,bootstrap,batch_size);
    }


    void init(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        ShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_opt, tree_init>::init_trees(X,Y,n_trees,bootstrap,batch_size);
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        return ShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_opt, tree_init>::predict_proba(X);
    }
    
    std::vector<internal_t> weights() const {
        return ShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_opt, tree_init>::weights();
    }

    unsigned int num_trees()const {
        return ShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_opt, tree_init>::num_trees();
    }

    unsigned int num_bytes()const {
        return ShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_opt, tree_init>::num_bytes();
    }
    
    unsigned int num_nodes() const {
        return ShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::NONE, tree_opt, tree_init>::num_nodes();
    }
};

#endif