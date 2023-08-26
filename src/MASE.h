#pragma once

#include <vector>

#include "Datatypes.h"
#include "Matrix.h"
#include "DecisionTree.h"
#include "TreeEnsemble.h"

template <typename data_t, INIT tree_init>
class MASE : public TreeEnsemble<data_t, tree_init> {
    
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
        unsigned int max_depth,
        unsigned int max_features,
        std::string loss = "mse",
        std::string optimizer = "adam",
        internal_t step_size = 0.1,
        unsigned long seed = 12345,
        unsigned int burnin_steps = 0,
        unsigned int n_trees = 32, 
        unsigned int n_worker = 8, 
        unsigned int n_rounds = 5,
        unsigned int batch_size = 0,
        bool bootstrap = true, 
        unsigned int init_tree_size = 0
    ) : TreeEnsemble<data_t, tree_init>(n_classes, nullptr, nullptr, seed , false, nullptr, nullptr, std::nullopt, 0, std::nullopt, 0), n_trees(n_trees), n_rounds(n_rounds), batch_size(batch_size), bootstrap(bootstrap), burnin_steps(burnin_steps), n_worker(n_worker), init_tree_size(init_tree_size) {

        if (loss == "mse" || loss == "MSE") {
            this->loss = std::make_unique<MSE>();
        } else if (loss == "CrossEntropy" || loss == "CROSSENTROPY") {
            this->loss = std::make_unique<CrossEntropy>();
        } else {
            std::runtime_error("Received a parameter that I did not understand. I understand loss = {mse, CrossEntropy}, but you gave me " + loss);
        }

        if (optimizer == "adam" || optimizer == "ADAM") {
            this->the_tree_optimizer = std::make_unique<Adam>(step_size);
        } else if (optimizer == "sgd" || optimizer == "SGD") {
            this->the_tree_optimizer = std::make_unique<SGD>(step_size);
        } else {
            std::runtime_error("Received a parameter that I did not understand. I understand optimizer = {adam, sgd}, but you gave me " + optimizer);
        }
    }

    MASE(
        unsigned int n_classes, 
        unsigned int max_depth, 
        unsigned int max_features, 
        Loss const & loss,
        Optimizer const & optimizer,
        unsigned long seed = 12345,
        unsigned int burnin_steps = 0,
        unsigned int n_trees = 32, 
        unsigned int n_worker = 8, 
        unsigned int n_rounds = 5,
        unsigned int batch_size = 0,
        bool bootstrap = true, 
        unsigned int init_tree_size = 0
    ) : TreeEnsemble<data_t, tree_init>(n_classes, max_depth, max_features, nullptr, seed , false, nullptr, nullptr, std::nullopt, 0, std::nullopt, 0), n_trees(n_trees), n_rounds(n_rounds), batch_size(batch_size), bootstrap(bootstrap), burnin_steps(burnin_steps), n_worker(n_worker), init_tree_size(init_tree_size) { 
        this->loss = loss.clone();
        this->the_tree_optimizer = optimizer.clone();
    }

    MASE(
        unsigned int n_classes, 
        unsigned int max_depth, 
        unsigned int max_features, 
        std::function< internal_t(std::vector<unsigned int> const &, std::vector<unsigned int> const &)> score,
        Loss const & loss,
        Optimizer const & optimizer,
        unsigned long seed = 12345,
        unsigned int burnin_steps = 0,
        unsigned int n_trees = 32, 
        unsigned int n_worker = 8, 
        unsigned int n_rounds = 5,
        unsigned int batch_size = 0,
        bool bootstrap = true, 
        unsigned int init_tree_size = 0
    ) : TreeEnsemble<data_t, tree_init>(n_classes, max_depth, max_features, score, nullptr, seed , false, nullptr, nullptr, std::nullopt, 0, std::nullopt, 0), n_trees(n_trees), n_rounds(n_rounds), batch_size(batch_size), bootstrap(bootstrap), burnin_steps(burnin_steps), n_worker(n_worker), init_tree_size(init_tree_size) { 
        this->loss = loss.clone();
        this->tree_optimizer = optimizer.clone();
    }

    // Copy constructor
    // MASE(const MASE& other) 
    //     : TreeEnsemble<data_t>(other.n_classes, nullptr, nullptr, other.seed , other.normalize_weights, nullptr, nullptr, other.ensemble_regularizer, other.l_ensemble_reg, other.tree_regularizer, other.l_tree_reg), n_trees(other.n_trees), n_rounds(other.n_rounds), batch_size(other.batch_size), bootstrap(other.bootstrap), burnin_steps(other.burnin_steps), n_worker(other.n_worker), init_tree_size(other.init_tree_size) {
    //     this->the_tree = other.the_tree->clone(other.seed);
    //     this->loss = other.loss->clone();
    //     this->the_tree_optimizer = other.the_tree_optimizer ? other.the_tree_optimizer->clone() : nullptr;
    //     this->weight_optimizer = other.weight_optimizer ? other.weight_optimizer->clone() : nullptr;

    //     this->_trees.reserve(other._trees.size());
    //     for (auto const & t : other._trees) {
    //         this->_trees.push_back(t->clone());
    //     }
    //     this->_weights = other._weights;
    // }

    void fit(matrix2d<data_t> const &X, matrix1d<unsigned int> const &Y, unsigned int n_trees, unsigned int init_tree_size, unsigned int n_parallel, bool bootstrap, unsigned int batch_size, unsigned int n_rounds, unsigned int burnin_steps) {
        this->init_trees(X, Y, n_trees, bootstrap, init_tree_size);
        
        for (unsigned int i = 0; i < n_rounds; ++i) {
            next(X,Y,n_parallel,bootstrap,batch_size,burnin_steps);
            if (this->ensemble_regularizer.has_value()) {
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
        // std::vector<TreeEnsemble<data_t>> ses(n_parallel, *this);
        std::vector<MASE<data_t, tree_init>> ses(n_parallel, *this);

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
                    this->_trees[j]->leaves() = ses[k]._trees[j]->leaves();
                } else {
                    this->_weights[j] += ses[k]._weights[j];

                    for (unsigned int l = 0; l < ses[k]._trees[j]->leaves().size(); ++l) {
                       this-> _trees[j]->leaves()[l] += ses[k]._trees[j]->leaves()[l];
                    }
                }
            }
            this->_weights[j] /= n_parallel;
            std::transform(this->_trees[j]->leaves().begin(), this->_trees[j]->leaves().end(), this->_trees[j]->leaves().begin(), [n_parallel](auto& c){return 1.0/n_parallel*c;});
        }
    }

    void init(matrix2d<data_t> const &X, matrix1d<unsigned int> const & Y) {
        if (init_tree_size == 0 || init_tree_size > X.rows) {
            this->init_trees(X,Y,n_trees,bootstrap,X.rows);
        } else {
            this->init_trees(X,Y,n_trees,bootstrap,init_tree_size);
        }
    }
};