#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "Prime.h"

class PrimeAdaptor {
private:
    PrimeInterface<data_t> * model = nullptr;

public:
    PrimeAdaptor(
        unsigned int n_classes, 
        unsigned int max_depth,
        unsigned long seed,
        bool normalize_weights,
        const std::string loss,
        data_t step_size,
        const std::string step_size_mode,
        const std::string ensemble_regularizer,
        data_t l_ensemble_reg,
        const std::string tree_regularizer,
        data_t l_tree_reg,
        const std::string tree_init_mode, 
        const std::string tree_update_mode
    ) { 

        // Yeha this is ugly and there is probably clever way to do this with C++17/20, but this was quicker to code and it gets the job done.
        // Also, lets be real here: There is only a limited chance more init/next modes are added without much refactoring of the whole project
        if (tree_init_mode == "random" && tree_update_mode == "incremental") {
            model = new Prime<TREE_INIT::RANDOM, TREE_NEXT::INCREMENTAL, data_t>( n_classes, max_depth, seed, normalize_weights, LOSS::from_string(loss), step_size, from_string(step_size_mode), ENSEMBLE_REGULARIZER::from_string(ensemble_regularizer), l_ensemble_reg, TREE_REGULARIZER::from_string(tree_regularizer), l_tree_reg );
        } else if (tree_init_mode == "random" && tree_update_mode == "gradient") {
            model = new Prime<TREE_INIT::RANDOM, TREE_NEXT::GRADIENT, data_t>( n_classes, max_depth, seed, normalize_weights, LOSS::from_string(loss), step_size, from_string(step_size_mode), ENSEMBLE_REGULARIZER::from_string(ensemble_regularizer), l_ensemble_reg, TREE_REGULARIZER::from_string(tree_regularizer), l_tree_reg );
        } else if (tree_init_mode == "random" && tree_update_mode == "none") {
            model = new Prime<TREE_INIT::RANDOM, TREE_NEXT::NONE, data_t>( n_classes, max_depth, seed, normalize_weights, LOSS::from_string(loss), step_size, from_string(step_size_mode), ENSEMBLE_REGULARIZER::from_string(ensemble_regularizer), l_ensemble_reg, TREE_REGULARIZER::from_string(tree_regularizer), l_tree_reg );
        // } else if (tree_init_mode == "fully-random" && tree_update_mode == "incremental") {
        //     model = new Prime<TREE_INIT::FULLY_RANDOM, TREE_NEXT::INCREMENTAL, data_t>( n_classes, max_depth, seed, normalize_weights, LOSS::from_string(loss), step_size, from_string(step_size_mode), ENSEMBLE_REGULARIZER::from_string(ensemble_regularizer), l_ensemble_reg, TREE_REGULARIZER::from_string(tree_regularizer), l_tree_reg );
        // } else if (tree_init_mode == "fully-random" && tree_update_mode == "gradient") {
        //     model = new Prime<TREE_INIT::FULLY_RANDOM, TREE_NEXT::GRADIENT, data_t>( n_classes, max_depth, seed, normalize_weights, LOSS::from_string(loss), step_size, from_string(step_size_mode), ENSEMBLE_REGULARIZER::from_string(ensemble_regularizer), l_ensemble_reg, TREE_REGULARIZER::from_string(tree_regularizer), l_tree_reg );
        // } else if (tree_init_mode == "fully-random" && tree_update_mode == "none") {
        //     model = new Prime<TREE_INIT::FULLY_RANDOM, TREE_NEXT::NONE, data_t>( n_classes, max_depth, seed, normalize_weights, LOSS::from_string(loss), step_size, from_string(step_size_mode), ENSEMBLE_REGULARIZER::from_string(ensemble_regularizer), l_ensemble_reg, TREE_REGULARIZER::from_string(tree_regularizer), l_tree_reg );
        } else if (tree_init_mode == "train" && tree_update_mode == "incremental") {
            model = new Prime<TREE_INIT::TRAIN, TREE_NEXT::INCREMENTAL, data_t>( n_classes, max_depth, seed, normalize_weights, LOSS::from_string(loss), step_size, from_string(step_size_mode), ENSEMBLE_REGULARIZER::from_string(ensemble_regularizer), l_ensemble_reg, TREE_REGULARIZER::from_string(tree_regularizer), l_tree_reg );
        } else if (tree_init_mode == "train" && tree_update_mode == "gradient") {
            model = new Prime<TREE_INIT::TRAIN, TREE_NEXT::GRADIENT, data_t>( n_classes, max_depth, seed, normalize_weights, LOSS::from_string(loss), step_size, from_string(step_size_mode), ENSEMBLE_REGULARIZER::from_string(ensemble_regularizer), l_ensemble_reg, TREE_REGULARIZER::from_string(tree_regularizer), l_tree_reg );
        } else if (tree_init_mode == "train" && tree_update_mode == "none") {
            model = new Prime<TREE_INIT::TRAIN, TREE_NEXT::NONE, data_t>( n_classes, max_depth, seed, normalize_weights, LOSS::from_string(loss), step_size, from_string(step_size_mode), ENSEMBLE_REGULARIZER::from_string(ensemble_regularizer), l_ensemble_reg, TREE_REGULARIZER::from_string(tree_regularizer), l_tree_reg );
        } else {
            throw std::runtime_error("Currently only the two tree_init_mode {random, train} and the three tree_update_mode {incremental, none, gradient} are supported for trees, but you provided a combination of " + tree_init_mode + " and " + tree_update_mode);
        }
    }

    void add_tree(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, data_t weight) {
        if (model != nullptr) {
            model->add_tree(X,Y,weight);
        }
    }

    void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            model->next(X,Y);
        }
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        if (model != nullptr) {
            return model->predict_proba(X);
        } else {
            // TODO Add defaults here? 
            return std::vector<std::vector<data_t>>();
        }
    }
    
    ~PrimeAdaptor() {
        if (model != nullptr) {
            delete model;
        }
    }

    std::vector<data_t> weights() const {
        if (model != nullptr) {
            return model->weights();
        } else {
            return std::vector<data_t>();
        }
    }

    unsigned int num_trees() const {
        if (model != nullptr) {
            return model->num_trees();
        } else {
            return 0;
        }
    }

    unsigned int num_bytes() const {
        if (model != nullptr) {
            return model->num_bytes();
        } else {
            return 0;
        }
    }

    unsigned int num_nodes() const {
        if (model != nullptr) {
            return model->num_nodes();
        } else {
            return 0;
        }
    }
};

class TreeAdaptor {
private:
    TreeInterface<data_t> * model = nullptr;

public:
    TreeAdaptor(
        unsigned int max_depth, 
        unsigned int n_classes, 
        unsigned long seed, 
        std::vector<std::vector<data_t>> const &X, 
        std::vector<unsigned int> const &Y,
        const std::string tree_init_mode, 
        const std::string tree_update_mode
    ) { 

        // Yeha this is ugly and there is probably clever way to do this with C++17/20, but this was quicker to code and it gets the job done.
        // Also, lets be real here: There is only a limited chance more init/next modes are added without much refactoring of the whole project
        if (tree_init_mode == "random" && tree_update_mode == "incremental") {
            model = new Tree<TREE_INIT::RANDOM, TREE_NEXT::INCREMENTAL, data_t>(max_depth, n_classes, seed, X, Y);
        } else if (tree_init_mode == "random" && tree_update_mode == "gradient") {
            model = new Tree<TREE_INIT::RANDOM, TREE_NEXT::GRADIENT, data_t>(max_depth, n_classes, seed, X, Y);
        } else if (tree_init_mode == "random" && tree_update_mode == "none") {
            model = new Tree<TREE_INIT::RANDOM, TREE_NEXT::NONE, data_t>(max_depth, n_classes, seed, X, Y);
        } else if (tree_init_mode == "train" && tree_update_mode == "incremental") {
            model = new Tree<TREE_INIT::TRAIN, TREE_NEXT::INCREMENTAL, data_t>(max_depth, n_classes, seed, X, Y);
        } else if (tree_init_mode == "train" && tree_update_mode == "gradient") {
            model = new Tree<TREE_INIT::TRAIN, TREE_NEXT::GRADIENT, data_t>(max_depth, n_classes, seed, X, Y);
        } else if (tree_init_mode == "train" && tree_update_mode == "none") {
            model = new Tree<TREE_INIT::TRAIN, TREE_NEXT::NONE, data_t>(max_depth, n_classes, seed, X, Y);
        } else {
            throw std::runtime_error("Currently only the two tree_init_mode {random, train} and the three tree_update_mode {incremental, none, gradient} are supported for trees, but you provided a combination of " + tree_init_mode + " and " + tree_update_mode);
        }
    }

    void next(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, std::vector<std::vector<data_t>> const &tree_grad, data_t step_size) {
        if (model != nullptr) {
            model->next(X, Y, tree_grad, step_size);
        }
    }

    std::vector<std::vector<data_t>> predict_proba(std::vector<std::vector<data_t>> const &X) {
        if (model != nullptr) {
            return model->predict_proba(X);
        } else {
            // TODO Add defaults here? 
            return std::vector<std::vector<data_t>>();
        }
    }
    
    ~TreeAdaptor() {
        if (model != nullptr) {
            delete model;
        }
    }

    unsigned int num_bytes() const {
        if (model != nullptr) {
            return model->num_bytes();
        } else {
            return 0;
        }
    }

    unsigned int num_nodes() const {
        if (model != nullptr) {
            return model->num_nodes();
        } else {
            return 0;
        }
    }
};

namespace py = pybind11;
PYBIND11_MODULE(CPrimeBindings, m) {

py::class_<PrimeAdaptor>(m, "CPrimeBindings")
    .def(py::init<unsigned int, unsigned int,unsigned long, bool, std::string, data_t, std::string, std::string, data_t, std::string, data_t, std::string, std::string>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("normalize_weights"), py::arg("loss"), py::arg("step_size"), py::arg("step_size_mode"), py::arg("ensemble_regularizer"), py::arg("l_ensemble_reg"), py::arg("tree_regularizer"), py::arg("l_tree_reg"), py::arg("tree_init_mode"), py::arg("tree_update_mode"))
    .def ("next", &PrimeAdaptor::next, py::arg("X"), py::arg("Y"))
    .def ("add_tree", &PrimeAdaptor::add_tree, py::arg("X"), py::arg("Y"), py::arg("weight"))
    .def ("num_trees", &PrimeAdaptor::num_trees)
    .def ("num_bytes", &PrimeAdaptor::num_bytes)
    .def ("num_nodes", &PrimeAdaptor::num_nodes)
    .def ("weights", &PrimeAdaptor::weights)
    .def ("predict_proba", &PrimeAdaptor::predict_proba, py::arg("X")
);

py::class_<TreeAdaptor>(m, "CTreeBindings")
    .def(py::init<unsigned int, unsigned int,unsigned long, std::vector<std::vector<data_t>>, std::vector<unsigned int>, std::string, std::string>(), py::arg("max_depth"), py::arg("n_classes"), py::arg("seed"), py::arg("X"), py::arg("Y"), py::arg("tree_init_mode"), py::arg("tree_update_mode"))
    .def ("next", &TreeAdaptor::next, py::arg("X"), py::arg("Y"), py::arg("tree_grad"), py::arg("step_size"))
    .def ("num_bytes", &TreeAdaptor::num_bytes)
    .def ("num_nodes", &TreeAdaptor::num_nodes)
    .def ("predict_proba", &TreeAdaptor::predict_proba, py::arg("X")
);

}