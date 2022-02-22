#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "ShrubEnsemble.h"
#include "OnlineShrubEnsemble.h"
#include "MAShrubEnsemble.h"
#include "GAShrubEnsemble.h"

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

        if (tree_init_mode == "random" && optimizer == "sgd" && loss == "mse") {
            model = new OnlineShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::RANDOM>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else if (tree_init_mode == "random" && optimizer == "adam" && loss == "mse") {
            model = new OnlineShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::RANDOM>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else if (tree_init_mode == "train" && optimizer == "sgd" && loss == "mse") {
            model = new OnlineShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::TRAIN>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else if (tree_init_mode == "train" && optimizer == "adam" && loss == "mse") {
            model = new OnlineShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::TRAIN>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else if (tree_init_mode == "random" && optimizer == "sgd" && loss == "cross-entropy") {
            model = new OnlineShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::RANDOM>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else if (tree_init_mode == "random" && optimizer == "adam" && loss == "cross-entropy") {
            model = new OnlineShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::RANDOM>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else if (tree_init_mode == "train" && optimizer == "sgd" && loss == "cross-entropy") {
            model = new OnlineShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::TRAIN>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else if (tree_init_mode == "train" && optimizer == "adam" && loss == "cross-entropy") {
            model = new OnlineShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::TRAIN>( n_classes, max_depth, seed, normalize_weights, burnin_steps, max_features, step_size, ENSEMBLE_REGULARIZER::from_string(regularizer), l_reg, TREE_REGULARIZER::from_string("none"), 0.0 );
        } else {
            throw std::runtime_error("Currently only the two tree_init_mode {random, train} and the two optimizer modes {adam, sgd} and the two losses {mse, cross-entropy} are supported for OnlineShrubEnsemble, but you provided a combination of " + tree_init_mode + " and " + optimizer + " and " + loss);
        }
    }

    void init(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y, unsigned int n_trees, bool bootstrap, unsigned int batch_size) {
        if (model != nullptr) {
            model->init(X,Y,n_trees,bootstrap,batch_size);
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

class GAShrubEnsembleAdaptor {
private:
    GAShrubEnsembleInterface * model = nullptr;

public:
    GAShrubEnsembleAdaptor(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        unsigned int max_features = 0,
        LOSS::TYPE loss = LOSS::TYPE::MSE, 
        internal_t step_size = 1e-2,
        OPTIMIZER::OPTIMIZER_TYPE optimizer = OPTIMIZER::OPTIMIZER_TYPE::SGD, 
        TREE_INIT tree_init_mode = TREE_INIT::TRAIN,
        unsigned int n_trees = 32, 
        unsigned int n_batches = 5,
        unsigned int n_rounds = 5,
        unsigned int init_batch_size = 0,
        bool bootstrap = true
    ) { 

        if (tree_init_mode == TREE_INIT::RANDOM && optimizer == OPTIMIZER::OPTIMIZER_TYPE::SGD && loss == LOSS::TYPE::MSE) {
            model = new GAShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::RANDOM>( n_classes, max_depth, seed,  max_features, step_size, n_trees, n_batches, n_rounds, init_batch_size, bootstrap );
        } else if (tree_init_mode == TREE_INIT::RANDOM && optimizer == OPTIMIZER::OPTIMIZER_TYPE::ADAM && loss == LOSS::TYPE::MSE) {
            model = new GAShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::RANDOM>( n_classes, max_depth, seed,  max_features, step_size, n_trees, n_batches, n_rounds, init_batch_size, bootstrap );
        } else if (tree_init_mode == TREE_INIT::TRAIN && optimizer == OPTIMIZER::OPTIMIZER_TYPE::SGD && loss == LOSS::TYPE::MSE) {
            model = new GAShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::TRAIN>( n_classes, max_depth, seed,  max_features, step_size, n_trees, n_batches, n_rounds, init_batch_size, bootstrap );
        } else if (tree_init_mode == TREE_INIT::TRAIN && optimizer == OPTIMIZER::OPTIMIZER_TYPE::ADAM && loss == LOSS::TYPE::MSE) {
            model = new GAShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::TRAIN>( n_classes, max_depth, seed,  max_features, step_size, n_trees, n_batches, n_rounds, init_batch_size, bootstrap );
        } else if (tree_init_mode == TREE_INIT::RANDOM && optimizer == OPTIMIZER::OPTIMIZER_TYPE::SGD && loss == LOSS::TYPE::CROSS_ENTROPY) {
            model = new GAShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::RANDOM>( n_classes, max_depth, seed,  max_features, step_size, n_trees, n_batches, n_rounds, init_batch_size, bootstrap );
        } else if (tree_init_mode == TREE_INIT::RANDOM && optimizer == OPTIMIZER::OPTIMIZER_TYPE::ADAM && loss == LOSS::TYPE::CROSS_ENTROPY) {
            model = new GAShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::RANDOM>( n_classes, max_depth, seed,  max_features, step_size, n_trees, n_batches, n_rounds, init_batch_size, bootstrap );
        } else if (tree_init_mode == TREE_INIT::TRAIN && optimizer == OPTIMIZER::OPTIMIZER_TYPE::SGD && loss == LOSS::TYPE::CROSS_ENTROPY) {
            model = new GAShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::TRAIN>( n_classes, max_depth, seed,  max_features, step_size, n_trees, n_batches, n_rounds, init_batch_size, bootstrap );
        } else if (tree_init_mode == TREE_INIT::TRAIN && optimizer == OPTIMIZER::OPTIMIZER_TYPE::ADAM && loss == LOSS::TYPE::CROSS_ENTROPY) {
            model = new GAShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::TRAIN>( n_classes, max_depth, seed,  max_features, step_size, n_trees, n_batches, n_rounds, init_batch_size, bootstrap );
        } 
    }

    GAShrubEnsembleAdaptor(
        unsigned int n_classes, 
        unsigned int max_depth = 5,
        unsigned long seed = 12345,
        unsigned int max_features = 0,
        const std::string loss = "mse",
        internal_t step_size = 1e-2,
        const std::string optimizer = "mse",
        const std::string tree_init_mode = "train",
        unsigned int n_trees = 32, 
        unsigned int n_batches = 5,
        unsigned int n_rounds = 5,
        unsigned int init_batch_size = 0,
        bool bootstrap = true
    ) { 
        LOSS::TYPE lt = LOSS::from_string(loss);
        OPTIMIZER::OPTIMIZER_TYPE ot = OPTIMIZER::optimizer_from_string(optimizer);
        TREE_INIT ti = tree_init_from_string(tree_init_mode);
        
        GAShrubEnsembleAdaptor(n_classes, max_depth, seed, max_features, lt, step_size, ot, ti, n_trees, n_batches, n_rounds, init_batch_size, bootstrap);
    }

    GAShrubEnsembleAdaptor(std::vector<unsigned char> &ga_string) {
        
    }

    std::vector<unsigned char> to_string() {
        std::vector<unsigned char> ga_string;

        serialize(model->n_classes, ga_string);
        serialize(model->max_depth, ga_string);
        serialize(model->seed, ga_string);
        serialize(model->max_features, ga_string);
        serialize(model->lt, ga_string);
        serialize(model->step_size, ga_string);
        serialize(model->ot, ga_string);
        serialize(model->ti, ga_string);
        serialize(model->n_trees, ga_string);
        serialize(model->n_batches, ga_string);
        serialize(model->n_rounds, ga_string);
        serialize(model->init_batch_size, ga_string);
        serialize(model->bootstrap, ga_string);

        return ga_string;
    }
    
    void init(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            model->init(X,Y);
        }
    }

    void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            model->fit(X,Y);
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
    
    ~GAShrubEnsembleAdaptor() {
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

/* MAShrubEnsembleAdaptor takes care of serialization and de-serialization. 
* The interface gives us also a list trees / weights + restoring? 
* The tree interface also gives us a lost of nodes / preds + restoring? 
*/

class MAShrubEnsembleAdaptor {
private:
    MAShrubEnsembleInterface * model = nullptr;

public:
    MAShrubEnsembleAdaptor(
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
        unsigned int n_parallel = 8, 
        unsigned int n_rounds = 5,
        unsigned int batch_size = 0,
        bool bootstrap = true
    ) { 
        if (tree_init_mode == "random" && optimizer == "sgd" && loss == "mse") {
            model = new MAShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::RANDOM>( n_classes, max_depth, seed, burnin_steps, max_features, step_size, n_trees, n_parallel, n_rounds, batch_size, bootstrap );
        } else if (tree_init_mode == "random" && optimizer == "adam" && loss == "mse") {
            model = new MAShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::RANDOM>( n_classes, max_depth, seed, burnin_steps, max_features, step_size, n_trees, n_parallel, n_rounds, batch_size, bootstrap );
        } else if (tree_init_mode == "train" && optimizer == "sgd" && loss == "mse") {
            model = new MAShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::TRAIN>( n_classes, max_depth, seed, burnin_steps, max_features, step_size, n_trees, n_parallel, n_rounds, batch_size, bootstrap );
        } else if (tree_init_mode == "train" && optimizer == "adam" && loss == "mse") {
            model = new MAShrubEnsemble<LOSS::TYPE::MSE, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::TRAIN>( n_classes, max_depth, seed, burnin_steps, max_features, step_size, n_trees, n_parallel, n_rounds, batch_size, bootstrap );
        } else if (tree_init_mode == "random" && optimizer == "sgd" && loss == "cross-entropy") {
            model = new MAShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::RANDOM>( n_classes, max_depth, seed, burnin_steps, max_features, step_size, n_trees, n_parallel, n_rounds, batch_size, bootstrap );
        } else if (tree_init_mode == "random" && optimizer == "adam" && loss == "cross-entropy") {
            model = new MAShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::RANDOM>( n_classes, max_depth, seed, burnin_steps, max_features, step_size, n_trees, n_parallel, n_rounds, batch_size, bootstrap );
        } else if (tree_init_mode == "train" && optimizer == "sgd" && loss == "cross-entropy") {
            model = new MAShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::SGD,TREE_INIT::TRAIN>( n_classes, max_depth, seed, burnin_steps, max_features, step_size, n_trees, n_parallel, n_rounds, batch_size, bootstrap );
        } else if (tree_init_mode == "train" && optimizer == "adam" && loss == "cross-entropy") {
            model = new MAShrubEnsemble<LOSS::TYPE::CROSS_ENTROPY, OPTIMIZER::OPTIMIZER_TYPE::ADAM,TREE_INIT::TRAIN>( n_classes, max_depth, seed, burnin_steps, max_features, step_size, n_trees, n_parallel, n_rounds, batch_size, bootstrap );
        } else {
            throw std::runtime_error("Currently only the two tree_init_mode {random, train} and the two optimizer modes {adam, sgd} are supported for Shrubes, but you provided a combination of " + tree_init_mode + " and " + optimizer);
        }
    }

    void init(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            model->init(X,Y);
        }
    }

    void fit(std::vector<std::vector<data_t>> const &X, std::vector<unsigned int> const &Y) {
        if (model != nullptr) {
            model->fit(X,Y);
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
    
    ~MAShrubEnsembleAdaptor() {
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
    .def ("init", &OnlineShrubEnsembleAdaptor::init, py::arg("X"), py::arg("Y"), py::arg("n_trees"), py::arg("bootstrap"), py::arg("batch_size"))
    .def ("next", &OnlineShrubEnsembleAdaptor::next, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &OnlineShrubEnsembleAdaptor::num_trees)
    .def ("num_bytes", &OnlineShrubEnsembleAdaptor::num_bytes)
    .def ("num_nodes", &OnlineShrubEnsembleAdaptor::num_nodes)
    .def ("weights", &OnlineShrubEnsembleAdaptor::weights)
    .def ("predict_proba", &OnlineShrubEnsembleAdaptor::predict_proba, py::arg("X")
);

py::class_<MAShrubEnsembleAdaptor>(m, "CMAShrubEnsembleBindings")
    .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, bool>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("burnin_steps"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_parallel"), py::arg("n_rounds"), py::arg("batch_size"), py::arg("bootstrap"))
    .def ("init", &MAShrubEnsembleAdaptor::init, py::arg("X"), py::arg("Y"))
    .def ("fit", &MAShrubEnsembleAdaptor::fit, py::arg("X"), py::arg("Y"))
    .def ("next", &MAShrubEnsembleAdaptor::next, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &MAShrubEnsembleAdaptor::num_trees)
    .def ("num_bytes", &MAShrubEnsembleAdaptor::num_bytes)
    .def ("num_nodes", &MAShrubEnsembleAdaptor::num_nodes)
    .def ("weights", &MAShrubEnsembleAdaptor::weights)
    .def ("predict_proba", &MAShrubEnsembleAdaptor::predict_proba, py::arg("X")
);

py::class_<GAShrubEnsembleAdaptor>(m, "CGAShrubEnsembleBindings")
    .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, unsigned int, bool>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_batchs"), py::arg("n_rounds"),py::arg("init_batch_size"), py::arg("bootstrap"))
    .def ("init", &GAShrubEnsembleAdaptor::init, py::arg("X"), py::arg("Y"))
    .def ("fit", &GAShrubEnsembleAdaptor::fit, py::arg("X"), py::arg("Y"))
    .def ("next", &GAShrubEnsembleAdaptor::next, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &GAShrubEnsembleAdaptor::num_trees)
    .def ("num_bytes", &GAShrubEnsembleAdaptor::num_bytes)
    .def ("num_nodes", &GAShrubEnsembleAdaptor::num_nodes)
    .def ("weights", &GAShrubEnsembleAdaptor::weights)
    .def ("predict_proba", &GAShrubEnsembleAdaptor::predict_proba, py::arg("X")
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