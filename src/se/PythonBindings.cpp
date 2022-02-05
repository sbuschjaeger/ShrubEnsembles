#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "ShrubEnsemble.h"
#include "OnlineShrubEnsemble.h"
#include "DistributedShrubEnsemble.h"

class OnlineShrubEnsembleAdaptor {
private:
    OnlineShrubEnsembleInterface * model = nullptr;

public:
    OnlineShrubEnsembleAdaptor(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        bool normalize_weights = true,
        unsigned int burnin_steps = 0,
        unsigned int max_features = 0,
        const std::string loss = "mse",
        internal_t step_size = 1e-2,
        const std::string optimizer = "sgd", 
        const std::string tree_init_mode = "train", 
        const std::string regularizer = "none",
        internal_t l_reg = 0
    ) { 
        if (tree_init_mode == "random" && optimizer == "sgd") {
            model = new OnlineShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::RANDOM>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, LOSS::from_string(loss), step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else if (tree_init_mode == "random" && optimizer == "adam") {
            model = new OnlineShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::RANDOM>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, LOSS::from_string(loss), step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else if (tree_init_mode == "train" && optimizer == "sgd") {
            model = new OnlineShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::TRAIN>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, LOSS::from_string(loss), step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else if (tree_init_mode == "train" && optimizer == "adam") {
            model = new OnlineShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::TRAIN>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, LOSS::from_string(loss), step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else {
            throw std::runtime_error("Currently only the two tree_init_mode {random, train} and the two optimizer modes {adam, sgd} are supported for Shrubes, but you provided a combination of " + tree_init_mode + " and " + optimizer);
        }
    }

    void init_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size) {
        if (model != nullptr) {
            model->init_trees(X,Y,n_trees,bootstrap,batch_size);
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
    
    ~OnlineShrubEnsembleAdaptor() {
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

class DistributedShrubEnsembleAdaptor {
private:
    DistributedShrubEnsembleInterface * model = nullptr;

public:
    DistributedShrubEnsembleAdaptor(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        unsigned int burnin_steps = 0,
        unsigned int max_features = 0,
        const std::string loss = "mse",
        internal_t step_size = 1e-2,
        const std::string optimizer = "mse",
        const std::string tree_init_mode = "train",
        unsigned int n_trees = 32, 
        unsigned int n_rounds = 5,
        unsigned int batch_size = 0,
        bool bootstrap = true
    ) { 
        if (tree_init_mode == "random" && optimizer == "sgd") {
            model = new DistributedShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::RANDOM>( n_classes, max_depth, seed, burnin_steps, max_features, LOSS::from_string(loss), step_size, n_trees, n_rounds, batch_size, bootstrap );
        } else if (tree_init_mode == "random" && optimizer == "adam") {
            model = new DistributedShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::RANDOM>( n_classes, max_depth, seed, burnin_steps, max_features, LOSS::from_string(loss), step_size, n_trees, n_rounds, batch_size, bootstrap );
        } else if (tree_init_mode == "train" && optimizer == "sgd") {
            model = new DistributedShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::TRAIN>( n_classes, max_depth, seed, burnin_steps, max_features, LOSS::from_string(loss), step_size, n_trees, n_rounds, batch_size, bootstrap );
        } else if (tree_init_mode == "train" && optimizer == "adam") {
            model = new DistributedShrubEnsemble<OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::TRAIN>( n_classes, max_depth, seed, burnin_steps, max_features, LOSS::from_string(loss), step_size, n_trees, n_rounds, batch_size, bootstrap );
        } else {
            throw std::runtime_error("Currently only the two tree_init_mode {random, train} and the two optimizer modes {adam, sgd} are supported for Shrubes, but you provided a combination of " + tree_init_mode + " and " + optimizer);
        }
    }

    void init_trees(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            model->init_trees(X,Y);
        }
    }

    void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            model->fit(X,Y);
        }
    }

    void next_distributed(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            model->next_distributed(X,Y);
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
    
    ~DistributedShrubEnsembleAdaptor() {
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
    TreeInterface * model = nullptr;

public:
    TreeAdaptor(
        unsigned int max_depth, 
        unsigned int n_classes, 
        unsigned int max_features,
        unsigned long seed, 
        internal_t step_size,
        const std::string tree_init_mode, 
        const std::string tree_optimizer
    ) { 

        // Yeha this is ugly and there is probably clever way to do this with C++17/20, but this was quicker to code and it gets the job done.
        // Also, lets be real here: There is only a limited chance more init/next modes are added without much refactoring of the whole project
        if (tree_init_mode == "random" && tree_optimizer == "sgd") {
            model = new Tree<TREE_INIT::RANDOM, OPTIMIZER::OPTIMIZER_TYPE::SGD>(n_classes,max_depth,max_features,seed,step_size);
        } else if (tree_init_mode == "random" && tree_optimizer == "adam") {
            model = new Tree<TREE_INIT::RANDOM, OPTIMIZER::OPTIMIZER_TYPE::ADAM>(n_classes,max_depth,max_features,seed,step_size);
        } else if (tree_init_mode == "random" && tree_optimizer == "none") {
            model = new Tree<TREE_INIT::RANDOM, OPTIMIZER::OPTIMIZER_TYPE::NONE>(n_classes,max_depth,max_features,seed,step_size);
        } else if (tree_init_mode == "train" && tree_optimizer == "sgd") {
            model = new Tree<TREE_INIT::TRAIN, OPTIMIZER::OPTIMIZER_TYPE::SGD>(n_classes,max_depth,max_features,seed,step_size);
        } else if (tree_init_mode == "train" && tree_optimizer == "adam") {
            model = new Tree<TREE_INIT::TRAIN, OPTIMIZER::OPTIMIZER_TYPE::ADAM>(n_classes,max_depth,max_features,seed,step_size);
        } else if (tree_init_mode == "train" && tree_optimizer == "none") {
            model = new Tree<TREE_INIT::TRAIN, OPTIMIZER::OPTIMIZER_TYPE::NONE>(n_classes,max_depth,max_features,seed,step_size);
        } else {
            throw std::runtime_error("Currently only the two tree_init_mode {random, train} and the three  optimizers {none,sgd,adam} are supported for trees, but you provided a combination of " + tree_init_mode + " and " + tree_optimizer);
        }
    }

    void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            model->fit(X, Y);
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
PYBIND11_MODULE(CShrubEnsembleBindings, m) {

py::class_<OnlineShrubEnsembleAdaptor>(m, "COnlineShrubEnsembleBindings")
    .def(py::init<unsigned int, unsigned int,unsigned long, bool, unsigned int, unsigned int, std::string, internal_t, std::string, std::string, std::string, internal_t>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("normalize_weights"), py::arg("burnin_steps"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("regularizer"), py::arg("l_reg"))
    .def ("init_trees", &OnlineShrubEnsembleAdaptor::init_trees, py::arg("X"), py::arg("Y"), py::arg("n_trees"), py::arg("bootstrap"), py::arg("batch_size"))
    .def ("next", &OnlineShrubEnsembleAdaptor::next, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &OnlineShrubEnsembleAdaptor::num_trees)
    .def ("num_bytes", &OnlineShrubEnsembleAdaptor::num_bytes)
    .def ("num_nodes", &OnlineShrubEnsembleAdaptor::num_nodes)
    .def ("weights", &OnlineShrubEnsembleAdaptor::weights)
    .def ("predict_proba", &OnlineShrubEnsembleAdaptor::predict_proba, py::arg("X")
);

py::class_<DistributedShrubEnsembleAdaptor>(m, "CDistributedShrubEnsembleBindings")
    .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, bool>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("burnin_steps"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_rounds"), py::arg("batch_size"), py::arg("bootstrap"))
    .def ("init_trees", &DistributedShrubEnsembleAdaptor::init_trees, py::arg("X"), py::arg("Y"))
    .def ("fit", &DistributedShrubEnsembleAdaptor::fit, py::arg("X"), py::arg("Y"))
    .def ("next_distributed", &DistributedShrubEnsembleAdaptor::next_distributed, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &DistributedShrubEnsembleAdaptor::num_trees)
    .def ("num_bytes", &DistributedShrubEnsembleAdaptor::num_bytes)
    .def ("num_nodes", &DistributedShrubEnsembleAdaptor::num_nodes)
    .def ("weights", &DistributedShrubEnsembleAdaptor::weights)
    .def ("predict_proba", &DistributedShrubEnsembleAdaptor::predict_proba, py::arg("X")
);

py::class_<TreeAdaptor>(m, "CTreeBindings")
    .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long, internal_t, std::string, std::string>(), py::arg("max_depth"), py::arg("n_classes"), py::arg("max_features"), py::arg("seed"), py::arg("step_size"), py::arg("tree_init_mode"), py::arg("tree_update_mode"))
    //.def ("next", &TreeAdaptor::next, py::arg("X"), py::arg("Y"), py::arg("tree_grad"))
    .def ("fit", &TreeAdaptor::fit, py::arg("X"), py::arg("Y"))
    .def ("num_bytes", &TreeAdaptor::num_bytes)
    .def ("num_nodes", &TreeAdaptor::num_nodes)
    .def ("predict_proba", &TreeAdaptor::predict_proba, py::arg("X")
);

}