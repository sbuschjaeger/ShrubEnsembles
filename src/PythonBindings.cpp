
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Datatypes.h"
#include "Matrix.h"

// #include "GASE.h"
// #include "MASE.h"
// #include "OSE.h"
#include "DecisionTree.h"

namespace py = pybind11;

namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<matrix1d<T>> {
private:
    py::array_t<T, py::array::c_style | py::array::forcecast> buf;

public:
    PYBIND11_TYPE_CASTER(matrix1d<T>, _("Matrix1d"));

    bool load(py::handle src, bool convert) {
        if ( !convert && !py::array_t<T>::check_(src) )
          return false;

        buf = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
        if ( !buf )
          return false;

        auto dims = buf.ndim();
        if ( dims != 1  )
          return false;

        value.data = std::unique_ptr<T []>(buf.mutable_data());
        value.dim = buf.shape()[0];
        value.has_ownership = false;

        return true;
    }

    static py::handle cast(const matrix1d<T>& src, py::return_value_policy, py::handle) {
        py::array_t<T> arr({src.dim}, {sizeof(T)}, src.data.get(), py::cast(src.has_ownership));
        return arr.release();
    }
};

template <typename T>
struct type_caster<matrix2d<T>> {
private:
    py::array_t<T, py::array::c_style | py::array::forcecast> buf;

public:
    PYBIND11_TYPE_CASTER(matrix2d<T>, _("Matrix2d"));

    bool load(py::handle src, bool convert)  {
        if ( !convert && !py::array_t<T>::check_(src) )
          return false;

        buf = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);
        if ( !buf )
          return false;

        auto dims = buf.ndim();
        if ( dims != 2  )
          return false;

        unsigned int n_rows = buf.shape()[0];
        unsigned int n_cols = buf.shape()[1];
        value.data = std::unique_ptr<T []>(buf.mutable_data());
        value.cols = n_cols;
        value.rows = n_rows;
        value.has_ownership = false;

        return true;
      }

    static py::handle cast(const matrix2d<T>& src, py::return_value_policy, py::handle) {
        py::array_t<T> arr({src.rows, src.cols}, {sizeof(T) * src.cols, sizeof(T)}, src.data.get(), py::cast(src.has_ownership));
        return arr.release();
    }
};

}  // namespace detail
}  // namespace pybind11

template <typename data_t, INIT tree_init>
void bindDecisionTree(py::module& m, const std::string& suffix) {
    using TreeType = DecisionTree<data_t, tree_init>;

    if constexpr(tree_init == INIT::CUSTOM) {
      py::class_<TreeType>(m, ("DecisionTree" + suffix).c_str(), py::module_local())
          .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long, std::function<internal_t(std::vector<unsigned int> const &, std::vector<unsigned int> const &)>>(),
              py::arg("n_classes"), py::arg("max_depth"), py::arg("max_features"),
              py::arg("seed"), py::arg("score_function"))
          .def("num_bytes", &TreeType::num_bytes)
          .def("num_nodes", &TreeType::num_nodes)
          .def("load", &TreeType::load, py::arg("nodes"))
          .def("predict_proba", py::overload_cast<matrix2d<data_t> const&>(&TreeType::predict_proba),
              py::arg("X"))
          .def("store", &TreeType::store)
          .def("fit", py::overload_cast<matrix2d<data_t> const&, matrix1d<unsigned int> const&,
                                      std::optional<std::reference_wrapper<const matrix1d<unsigned int>>>>(&TreeType::fit),
              py::arg("X"), py::arg("Y"), py::arg("indices") = py::none());
    } else {
      py::class_<TreeType>(m, ("DecisionTree" + suffix).c_str(), py::module_local())
          .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long>(),
              py::arg("n_classes"), py::arg("max_depth"), py::arg("max_features"),
              py::arg("seed"))
          .def("num_bytes", &TreeType::num_bytes)
          .def("num_nodes", &TreeType::num_nodes)
          .def("load", &TreeType::load, py::arg("nodes"))
          .def("predict_proba", py::overload_cast<matrix2d<data_t> const&>(&TreeType::predict_proba),
              py::arg("X"))
          .def("store", &TreeType::store)
          .def("fit", py::overload_cast<matrix2d<data_t> const&, matrix1d<unsigned int> const&,
                                      std::optional<std::reference_wrapper<const matrix1d<unsigned int>>>>(&TreeType::fit),
              py::arg("X"), py::arg("Y"), py::arg("indices") = py::none());
    }
}


// using MASE_T = MASE<internal_t>;

PYBIND11_MODULE(ShrubEnsemblesCPP, m) {
  bindDecisionTree<double, INIT::GINI>(m, "_DOUBLE_GINI");
  bindDecisionTree<double, INIT::RANDOM>(m, "_DOUBLE_RANDOM");
  bindDecisionTree<double, INIT::CUSTOM>(m, "_DOUBLE_CUSTOM");
  bindDecisionTree<int, INIT::GINI>(m, "_INT_GINI");
  bindDecisionTree<int, INIT::RANDOM>(m, "_INT_RANDOM");
  bindDecisionTree<int, INIT::CUSTOM>(m, "_INT_CUSTOM");

// py::class_<MASE_T>(m, "MASE")
//     .def(py::init<unsigned int, unsigned int, unsigned long, std::string, std::string, std::string, internal_t, unsigned long, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, unsigned int>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("max_featues"), py::arg("init"), py::arg("loss"), py::arg("optimizer"), py::arg("step_size"), py::arg("seed"),  py::arg("burnin_steps"), py::arg("n_trees"), py::arg("n_worker"), py::arg("n_rounds"), py::arg("batch_size"), py::arg("bootstrap"), py::arg("init_tree_size"))
//     .def ("init", &MASE_T::init, py::arg("X"), py::arg("Y"))
//     .def ("fit", py::overload_cast<matrix2d<internal_t> const&, matrix1d<unsigned int> const&>(&MASE_T::fit), py::arg("X"), py::arg("Y"))
//     .def ("next", py::overload_cast<matrix2d<internal_t> const&, matrix1d<unsigned int> const&>(&MASE_T::fit), py::arg("X"), py::arg("Y"))
//     .def ("predict_proba", &MASE_T::predict_proba, py::arg("X"))
//     .def ("num_nodes", &MASE_T::num_nodes)
//     .def ("num_bytes", &MASE_T::num_bytes)
//     .def ("load", &MASE_T::load, py::arg("new_nodes"), py::arg("new_weights"))
//     .def ("store", &MASE_T::store)
//     .def ("num_trees", &MASE_T::num_trees
// );

// py::class_<GASE>(m, "CGASE")
//     .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_worker"), py::arg("n_rounds"),py::arg("init_batch_size"), py::arg("batch_size"), py::arg("bootstrap"))
//     .def ("init", &GASE::init, py::arg("X"), py::arg("Y"))
//     .def ("fit", &GASE::fit, py::arg("X"), py::arg("Y"))
//     .def ("next", &GASE::next, py::arg("X"), py::arg("Y"))
//     // .def ("leafs", &PyGase::leafs)
//     // .def ("nodes", &PyGase::nodes)
//     .def ("predict_proba", &GASE::predict_proba, py::arg("X"))
//     .def ("num_nodes", &GASE::num_nodes)
//     .def ("num_bytes", &GASE::num_bytes)
//     .def ("load", &GASE::load, py::arg("new_nodes"), py::arg("new_leafs"), py::arg("new_weights"))
//     .def ("store", &GASE::store)
//     .def ("num_trees", &GASE::num_trees
// );

// py::class_<DecisionTreeClassifier>(m, "CDecisionTreeClassifier")
//     .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long, internal_t, std::string, std::string>(), py::arg("max_depth"), py::arg("n_classes"), py::arg("max_features"), py::arg("seed"), py::arg("step_size"), py::arg("tree_init_mode"), py::arg("tree_update_mode"))
//     //.def ("next", &TreeAdaptor::next, py::arg("X"), py::arg("Y"), py::arg("tree_grad"))
//     .def ("fit", &DecisionTreeClassifier::fit, py::arg("X"), py::arg("Y"))
//     .def ("num_bytes", &DecisionTreeClassifier::num_bytes)
//     .def ("num_nodes", &DecisionTreeClassifier::num_nodes)
//     .def ("load", &DecisionTreeClassifier::load, py::arg("new_nodes"), py::arg("new_leafs"))
//     .def ("store", &DecisionTreeClassifier::store)
//     .def ("predict_proba", &DecisionTreeClassifier::predict_proba, py::arg("X")
// );




// py::class_<DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::ZLIB, OPTIMIZER::NONE>>(m, "DistanceDecisionTreeTrainZlibNone")
//     .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long, internal_t, internal_t>(), py::arg("max_depth"), py::arg("n_classes"), py::arg("max_features"), py::arg("seed"), py::arg("lambda"), py::arg("step_size"))
//     .def("num_bytes", &DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::ZLIB, OPTIMIZER::NONE>::num_bytes)
//     .def("num_nodes", &DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::ZLIB, OPTIMIZER::NONE>::num_nodes)
//     .def("load", &DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::ZLIB, OPTIMIZER::NONE>::load, py::arg("nodes"))
//     .def("predict_proba", py::overload_cast<matrix2d<data_t> const &>(&DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::ZLIB, OPTIMIZER::NONE>::predict_proba), py::arg("nodes"))
//     .def("store", &DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::ZLIB, OPTIMIZER::NONE>::store)
//     .def("fit", py::overload_cast<matrix2d<data_t> const &, matrix1d<unsigned int> const &, std::optional<std::reference_wrapper<const matrix1d<unsigned int>>>>(&DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::ZLIB, OPTIMIZER::NONE>::fit), py::arg("X"), py::arg("Y"), py::arg("indices"))
//     .def("fit", py::overload_cast<matrix2d<data_t> const &, matrix1d<unsigned int> const &, matrix1d<unsigned int> const &, matrix2d<data_t> const &>(&DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::ZLIB, OPTIMIZER::NONE>::fit), py::arg("X"), py::arg("Y"), py::arg("indices"), py::arg("distance_matrix")
// );

// py::class_<DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::EUCLIDEAN, OPTIMIZER::NONE>>(m, "DistanceDecisionTreeTrainEuclideanNone")
//     .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long, internal_t, internal_t>(), py::arg("max_depth"), py::arg("n_classes"), py::arg("max_features"), py::arg("seed"), py::arg("lambda"), py::arg("step_size"))
//     .def("num_bytes", &DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::EUCLIDEAN, OPTIMIZER::NONE>::num_bytes)
//     .def("num_nodes", &DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::EUCLIDEAN, OPTIMIZER::NONE>::num_nodes)
//     .def("load", &DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::EUCLIDEAN, OPTIMIZER::NONE>::load, py::arg("nodes"))
//     .def("predict_proba", py::overload_cast<matrix2d<data_t> const &>(&DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::EUCLIDEAN, OPTIMIZER::NONE>::predict_proba), py::arg("nodes"))
//     .def("store", &DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::EUCLIDEAN, OPTIMIZER::NONE>::store)
//     .def("fit", py::overload_cast<matrix2d<data_t> const &, matrix1d<unsigned int> const &, std::optional<std::reference_wrapper<const matrix1d<unsigned int>>>>(&DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::EUCLIDEAN, OPTIMIZER::NONE>::fit), py::arg("X"), py::arg("Y"), py::arg("indices"))
//     .def("fit", py::overload_cast<matrix2d<data_t> const &, matrix1d<unsigned int> const &, matrix1d<unsigned int> const &, matrix2d<data_t> const &>(&DistanceDecisionTree<DDT::TREE_INIT::TRAIN, DISTANCE::TYPES::EUCLIDEAN, OPTIMIZER::NONE>::fit), py::arg("X"), py::arg("Y"), py::arg("indices"), py::arg("distance_matrix")
// );

}