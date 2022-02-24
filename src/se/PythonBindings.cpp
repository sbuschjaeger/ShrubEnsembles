#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "DecisionTree.h"
#include "GASE.h"
#include "MASE.h"
#include "OSE.h"

namespace py = pybind11;
PYBIND11_MODULE(CShrubEnsembles, m) {

py::class_<OSE>(m, "COSE")
    .def(py::init<unsigned int, unsigned int,unsigned long, bool, unsigned int, unsigned int, std::string, internal_t, std::string, std::string, std::string, internal_t>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("normalize_weights"), py::arg("burnin_steps"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("regularizer"), py::arg("l_reg"))
    .def ("init", &OSE::init, py::arg("X"), py::arg("Y"), py::arg("n_trees"), py::arg("bootstrap"), py::arg("batch_size"))
    .def ("next", &OSE::next, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &OSE::num_trees)
    .def ("num_bytes", &OSE::num_bytes)
    .def ("num_nodes", &OSE::num_nodes)
    .def ("weights", &OSE::weights)
    .def ("predict_proba", &OSE::predict_proba, py::arg("X")
);

py::class_<MASE>(m, "CMASE")
    .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, unsigned int, bool>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("burnin_steps"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_parallel"), py::arg("n_rounds"), py::arg("batch_size"), py::arg("bootstrap"))
    .def ("init", &MASE::init, py::arg("X"), py::arg("Y"))
    .def ("fit", &MASE::fit, py::arg("X"), py::arg("Y"))
    .def ("next", &MASE::next, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &MASE::num_trees)
    .def ("num_bytes", &MASE::num_bytes)
    .def ("num_nodes", &MASE::num_nodes)
    .def ("weights", &MASE::weights)
    .def ("predict_proba", &MASE::predict_proba, py::arg("X"))
    .def ("set_weights", &MASE::set_weights, py::arg("new_weights"))
    .def ("set_leafs", &MASE::set_leafs, py::arg("new_leafs"))
    .def ("set_nodes", &MASE::set_nodes, py::arg("new_nodes")
);

py::class_<GASE>(m, "CGASE")
    .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, unsigned int, bool>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_batchs"), py::arg("n_rounds"),py::arg("init_batch_size"), py::arg("bootstrap"))
    .def ("init", &GASE::init, py::arg("X"), py::arg("Y"))
    .def ("fit", &GASE::fit, py::arg("X"), py::arg("Y"))
    .def ("next", &GASE::next, py::arg("X"), py::arg("Y"))
    .def ("num_trees", &GASE::num_trees)
    .def ("num_bytes", &GASE::num_bytes)
    .def ("num_nodes", &GASE::num_nodes)
    .def ("weights", &GASE::weights)
    .def ("predict_proba", &GASE::predict_proba, py::arg("X"))
    .def ("set_weights", &GASE::set_weights, py::arg("new_weights"))
    .def ("set_leafs", &GASE::set_leafs, py::arg("new_leafs"))
    .def ("set_nodes", &GASE::set_nodes, py::arg("new_nodes")
);

py::class_<DecisionTreeClassifier>(m, "CDecisionTreeClassifier")
    .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long, internal_t, std::string, std::string>(), py::arg("max_depth"), py::arg("n_classes"), py::arg("max_features"), py::arg("seed"), py::arg("step_size"), py::arg("tree_init_mode"), py::arg("tree_update_mode"))
    //.def ("next", &TreeAdaptor::next, py::arg("X"), py::arg("Y"), py::arg("tree_grad"))
    .def ("fit", &DecisionTreeClassifier::fit, py::arg("X"), py::arg("Y"))
    .def ("num_bytes", &DecisionTreeClassifier::num_bytes)
    .def ("num_nodes", &DecisionTreeClassifier::num_nodes)
    .def ("predict_proba", &DecisionTreeClassifier::predict_proba, py::arg("X"))
    .def ("set_leafs", &DecisionTreeClassifier::set_leafs, py::arg("new_leafs"))
    .def ("set_nodes", &DecisionTreeClassifier::set_nodes, py::arg("new_nodes")
);

}