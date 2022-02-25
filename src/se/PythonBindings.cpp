#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "DecisionTree.h"
#include "GASE.h"
#include "MASE.h"
#include "OSE.h"

// class PyDecisionTreeClassifier : public DecisionTreeClassifier {
// public:
//     using DecisionTreeClassifier::DecisionTreeClassifier;

//     void update_leafs(std::vector<internal_t> & new_leafs) const {
//         if (tree != nullptr) {
//             std::cout << "BEFORE MOVE" << std::endl;
//             tree->leafs() = std::move(new_leafs);
//             std::cout << "AFTER MOVE" << std::endl;
//         } else {
//             throw std::runtime_error("The internal object pointer in PyDecisionTreeClassifier was null. This should now happen!");
//         }
//     }

//     std::vector<internal_t> leafs() const {
//         if (tree != nullptr) {
//             return std::vector<internal_t>(tree->leafs());
//         } else {
//             throw std::runtime_error("The internal object pointer in PyDecisionTreeClassifier was null. This should now happen!");
//         }
//     }

//     void update_nodes(std::vector<internal_t> & new_nodes) const {
//         if (tree != nullptr) {
//             std::vector<Node> t_nodes;
//             for (unsigned int j = 0; j < new_nodes.size(); j += 6) {
//                 Node n;
//                 n.threshold = static_cast<data_t>(new_nodes[j]);
//                 n.feature = static_cast<unsigned int>(new_nodes[j+1]);
//                 n.left = static_cast<unsigned int>(new_nodes[j+2]);
//                 n.right = static_cast<unsigned int>(new_nodes[j+3]);
//                 n.left_is_leaf = new_nodes[j+4] == 0.0 ? false : true;
//                 n.right_is_leaf = new_nodes[j+5] == 0.0 ? false : true;
//                 t_nodes.push_back(n);
//             }

//             tree->nodes() = std::move(t_nodes);
//         } else {
//             throw std::runtime_error("The internal object pointer in PyMase was null. This should now happen!");
//         }
//     }

//     std::vector<internal_t> nodes() {
//         if (tree != nullptr) {
//             std::vector<internal_t> primitve_nodes(tree->nodes().size());

//             for (auto const &n : tree->nodes()) {
//                 primitve_nodes.push_back(static_cast<internal_t>(n.threshold));
//                 primitve_nodes.push_back(static_cast<internal_t>(n.feature));
//                 primitve_nodes.push_back(static_cast<internal_t>(n.left));
//                 primitve_nodes.push_back(static_cast<internal_t>(n.right));
//                 primitve_nodes.push_back(static_cast<internal_t>(n.left_is_leaf));
//                 primitve_nodes.push_back(static_cast<internal_t>(n.right_is_leaf));
//             }
//             return primitve_nodes;
//         } else {
//             throw std::runtime_error("The internal object pointer in PyMase was null. This should now happen!");
//         }
//     }

// };

// class PyMase : public MASE {
// public:
//     using MASE::MASE;

//     void update_parameters(std::vector<std::vector<internal_t>> & new_nodes, std::vector<std::vector<internal_t>> & new_leafs, std::vector<internal_t> & weights) {
        
//         model->trees().reset();
//         for (unsigned int i = 0; i < weights.size(); ++i) {
//             if (tree_init_mode == "random" && optimizer == "sgd") {
//                 model->trees().push_back(
//                     DecisionTree<DT::TREE_INIT::RANDOM, OPTIMIZER::OPTIMIZER_TYPE::SGD>(n_classes,max_depth, max_features, seed++, step_size)
//                 );
//             } else if (tree_init_mode == "random" && optimizer == "adam") {
//                 model->trees().push_back(
//                     DecisionTree<DT::TREE_INIT::RANDOM, OPTIMIZER::OPTIMIZER_TYPE::ADAM>(n_classes,max_depth, max_features, seed++, step_size)
//                 );
//             } else if (tree_init_mode == "train" && optimizer == "sgd") {
//                 model->trees().push_back(
//                     DecisionTree<DT::TREE_INIT::TRAIN, OPTIMIZER::OPTIMIZER_TYPE::SGD>(n_classes,max_depth, max_features, seed++, step_size)
//                 );
//             } else {
//                 model->trees().push_back(
//                     DecisionTree<DT::TREE_INIT::TRAIN, OPTIMIZER::OPTIMIZER_TYPE::ADAM>(n_classes,max_depth, max_features, seed++, step_size)
//                 );
//             }
//             model->trees().back().leafs() = std::move(new_leafs[i]);
//             model->trees().back().nodes() = std::move(new_nodes[i]);
//         }

//         model->weights() = new_weights;
//     }

//     std::vector<internal_t> weights() const {
//         if (model != nullptr) {
//             return std::vector<internal_t>(model->weights());
//         } else {
//             throw std::runtime_error("The internal object pointer in PyMase was null. This should now happen!");
//         }
//     }

//     void update_weights(std::vector<internal_t> & weights) const {
//         if (model != nullptr) {
//             model->weights() = std::move(weights);
//         } else {
//             throw std::runtime_error("The internal object pointer in PyMase was null. This should now happen!");
//         }
//     }

//     void update_leafs(std::vector<std::vector<internal_t>> & new_leafs) const {
//         if (model != nullptr) {
//             std::vector<Tree*> trees = model->trees();
//             for (unsigned int i = 0; i < new_leafs.size(); ++i) {
//                 trees.push_back(
//                     new DecisionTree<tree_init, tree_opt>(n_classes, max_depth, max_features, seed+i, step_size)
//                 );
//                 trees[i]->leafs() = std::move(new_leafs[i]);
//             }
//             std::cout << "AFTER MOVE" << std::endl;
//         } else {
//             throw std::runtime_error("The internal object pointer in PyMase was null. This should now happen!");
//         }
//     }

//     std::vector<std::vector<internal_t>> leafs() const {
//         if (model != nullptr) {
//             std::vector<Tree*> trees = model->trees();
//             std::vector<std::vector<internal_t>> all_leafs(trees.size());

//             for (unsigned int i = 0; i < trees.size(); ++i) {
//                 all_leafs[i] = trees[i]->leafs();
//             }

//             return all_leafs;
//         } else {
//             throw std::runtime_error("The internal object pointer in PyMase was null. This should now happen!");
//         }
//     }

//     void update_nodes(std::vector<std::vector<internal_t>> & new_nodes) const {
//         if (model != nullptr) {
//             std::vector<Tree*> trees = model->trees();
//             for (unsigned int i = 0; i < new_nodes.size(); ++i) {
//                 std::vector<Node> t_nodes;
//                 for (unsigned int j = 0; j < new_nodes[i].size(); j += 6) {
//                     Node n;
//                     n.threshold = static_cast<data_t>(new_nodes[i][j]);
//                     n.feature = static_cast<unsigned int>(new_nodes[i][j+1]);
//                     n.left = static_cast<unsigned int>(new_nodes[i][j+2]);
//                     n.right = static_cast<unsigned int>(new_nodes[i][j+3]);
//                     n.left_is_leaf = new_nodes[i][j+4] == 0.0 ? false : true;
//                     n.right_is_leaf = new_nodes[i][j+5] == 0.0 ? false : true;
//                     t_nodes.push_back(n);
//                 }

//                 trees[i]->nodes() = std::move(t_nodes);
//             }
//         } else {
//             throw std::runtime_error("The internal object pointer in PyMase was null. This should now happen!");
//         }
//     }

//     std::vector<std::vector<internal_t>> nodes() {
//         if (model != nullptr) {
//             std::vector<Tree*> trees = model->trees();
//             std::vector<std::vector<internal_t>> primitve_nodes(trees.size());

//             for (unsigned int i = 0; i < trees.size(); ++i) {
//                 for (auto const &n : trees[i]->nodes()) {
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.threshold));
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.feature));
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.left));
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.right));
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.left_is_leaf));
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.right_is_leaf));
//                 }
//             } 
//             return primitve_nodes;
//         } else {
//             throw std::runtime_error("The internal object pointer in PyMase was null. This should now happen!");
//         }
//     }
// };

// class PyGase : public GASE {
// public:
//     using GASE::GASE;

//     std::vector<internal_t> weights() const {
//         if (model != nullptr) {
//             return std::vector<internal_t>(model->weights());
//         } else {
//             throw std::runtime_error("The internal object pointer in PyGase was null. This should now happen!");
//         }
//     }

//     void update_weights(std::vector<internal_t> & weights) const {
//         if (model != nullptr) {
//             model->weights() = std::move(weights);
//         } else {
//             throw std::runtime_error("The internal object pointer in PyGase was null. This should now happen!");
//         }
//     }

//     void update_leafs(std::vector<std::vector<internal_t>> & new_leafs) const {
//         if (model != nullptr) {
//             std::vector<Tree*> trees = model->trees();
//             for (unsigned int i = 0; i < new_leafs.size(); ++i) {
//                 trees[i]->leafs() = std::move(new_leafs[i]);
//             }
//         } else {
//             throw std::runtime_error("The internal object pointer in PyGase was null. This should now happen!");
//         }
//     }

//     std::vector<std::vector<internal_t>> leafs() const {
//         if (model != nullptr) {
//             std::vector<Tree*> trees = model->trees();
//             std::vector<std::vector<internal_t>> all_leafs(trees.size());

//             for (unsigned int i = 0; i < trees.size(); ++i) {
//                 all_leafs[i] = trees[i]->leafs();
//             }

//             return all_leafs;
//         } else {
//             throw std::runtime_error("The internal object pointer in PyGase was null. This should now happen!");
//         }
//     }

//     void update_nodes(std::vector<std::vector<internal_t>> & new_nodes) const {
//         if (model != nullptr) {
//             std::vector<Tree*> trees = model->trees();
//             for (unsigned int i = 0; i < new_nodes.size(); ++i) {
//                 std::vector<Node> t_nodes;
//                 for (unsigned int j = 0; j < new_nodes[i].size(); j += 6) {
//                     Node n;
//                     n.threshold = static_cast<data_t>(new_nodes[i][j]);
//                     n.feature = static_cast<unsigned int>(new_nodes[i][j+1]);
//                     n.left = static_cast<unsigned int>(new_nodes[i][j+2]);
//                     n.right = static_cast<unsigned int>(new_nodes[i][j+3]);
//                     n.left_is_leaf = new_nodes[i][j+4] == 0.0 ? false : true;
//                     n.right_is_leaf = new_nodes[i][j+5] == 0.0 ? false : true;
//                     t_nodes.push_back(n);
//                 }

//                 trees[i]->nodes() = std::move(t_nodes);
//             }
//         } else {
//             throw std::runtime_error("The internal object pointer in PyGase was null. This should now happen!");
//         }
//     }

//     std::vector<std::vector<internal_t>> nodes() {
//         if (model != nullptr) {
//             std::vector<Tree*> trees = model->trees();
//             std::vector<std::vector<internal_t>> primitve_nodes(trees.size());

//             for (unsigned int i = 0; i < trees.size(); ++i) {
//                 for (auto const &n : trees[i]->nodes()) {
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.threshold));
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.feature));
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.left));
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.left));
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.left_is_leaf));
//                     primitve_nodes[i].push_back(static_cast<internal_t>(n.right_is_leaf));
//                 }
//             } 
//             return primitve_nodes;
//         } else {
//             throw std::runtime_error("The internal object pointer in PyMase was null. This should now happen!");
//         }
//     }
// };

namespace py = pybind11;
PYBIND11_MODULE(CShrubEnsembles, m) {

py::class_<OSE>(m, "COSE")
    .def(py::init<unsigned int, unsigned int,unsigned long, bool, unsigned int, unsigned int, std::string, internal_t, std::string, std::string, std::string, internal_t>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("normalize_weights"), py::arg("burnin_steps"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("regularizer"), py::arg("l_reg"))
    .def ("init", &OSE::init, py::arg("X"), py::arg("Y"), py::arg("n_trees"), py::arg("bootstrap"), py::arg("batch_size"))
    .def ("next", &OSE::next, py::arg("X"), py::arg("Y"))
    .def ("predict_proba", &OSE::predict_proba, py::arg("X"))
    .def ("num_nodes", &OSE::num_nodes)
    .def ("num_bytes", &OSE::num_bytes)
    .def ("num_trees", &OSE::num_trees
);

py::class_<MASE>(m, "CMASE")
    .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, unsigned int, bool>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("burnin_steps"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_parallel"), py::arg("n_rounds"), py::arg("batch_size"), py::arg("bootstrap"))
    .def ("init", &MASE::init, py::arg("X"), py::arg("Y"))
    .def ("fit", &MASE::fit, py::arg("X"), py::arg("Y"))
    .def ("next", &MASE::next, py::arg("X"), py::arg("Y"))
    .def ("predict_proba", &MASE::predict_proba, py::arg("X"))
    // .def ("leafs", &PyMase::leafs)
    // .def ("nodes", &PyMase::nodes)
    .def ("num_nodes", &MASE::num_nodes)
    .def ("num_bytes", &MASE::num_bytes)
    .def ("load", &MASE::load, py::arg("new_nodes"), py::arg("new_leafs"), py::arg("new_weights"))
    .def ("store", &MASE::store)
    // .def ("weights", &PyMase::weights)
    // .def ("update_weights", &PyMase::update_weights, py::arg("new_weights"))
    // .def ("update_nodes", &PyMase::update_nodes, py::arg("new_nodes"))
    // .def ("update_leafs", &PyMase::update_leafs, py::arg("update_leafs"))
    .def ("num_trees", &MASE::num_trees
);

py::class_<GASE>(m, "CGASE")
    .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, unsigned int, bool>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_batchs"), py::arg("n_rounds"),py::arg("init_batch_size"), py::arg("bootstrap"))
    .def ("init", &GASE::init, py::arg("X"), py::arg("Y"))
    .def ("fit", &GASE::fit, py::arg("X"), py::arg("Y"))
    .def ("next", &GASE::next, py::arg("X"), py::arg("Y"))
    // .def ("leafs", &PyGase::leafs)
    // .def ("nodes", &PyGase::nodes)
    .def ("predict_proba", &GASE::predict_proba, py::arg("X"))
    .def ("num_nodes", &GASE::num_nodes)
    .def ("num_bytes", &GASE::num_bytes)
    .def ("load", &GASE::load, py::arg("new_nodes"), py::arg("new_leafs"), py::arg("new_weights"))
    .def ("store", &GASE::store)
    // .def ("update_weights", &PyGase::update_weights, py::arg("new_weights"))
    // .def ("update_nodes", &PyGase::update_nodes, py::arg("new_nodes"))
    // .def ("update_nodes", &PyGase::update_nodes, py::arg("new_nodes"))
    // .def ("update_leafs", &PyGase::update_leafs, py::arg("update_leafs"))
    .def ("num_trees", &GASE::num_trees
);

py::class_<DecisionTreeClassifier>(m, "CDecisionTreeClassifier")
    .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long, internal_t, std::string, std::string>(), py::arg("max_depth"), py::arg("n_classes"), py::arg("max_features"), py::arg("seed"), py::arg("step_size"), py::arg("tree_init_mode"), py::arg("tree_update_mode"))
    //.def ("next", &TreeAdaptor::next, py::arg("X"), py::arg("Y"), py::arg("tree_grad"))
    .def ("fit", &DecisionTreeClassifier::fit, py::arg("X"), py::arg("Y"))
    .def ("num_bytes", &DecisionTreeClassifier::num_bytes)
    .def ("num_nodes", &DecisionTreeClassifier::num_nodes)
    .def ("load", &DecisionTreeClassifier::load, py::arg("new_nodes"), py::arg("new_leafs"))
    .def ("store", &DecisionTreeClassifier::store)
    .def ("predict_proba", &DecisionTreeClassifier::predict_proba, py::arg("X")
    // .def ("update_leafs", &DecisionTreeClassifier::update_leafs, py::arg("new_leafs"))
    // .def ("leafs", &DecisionTreeClassifier::leafs)
    // .def ("update_nodes", &DecisionTreeClassifier::update_nodes, py::arg("new_nodes"))
    // .def ("nodes", &PyDecisionTreeClassifier::nodes
);

}