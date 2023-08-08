
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Datatypes.h"
#include "Matrix.h"

#include "GASE.h"
#include "MASE.h"
#include "OSE.h"
#include "DecisionTree.h"
#include "DistanceDecisionTree.h"

namespace py = pybind11;

namespace pybind11 { namespace detail {
  template<> struct type_caster<matrix2d<data_t>>
  {
    private:
        py::array_t<data_t, py::array::c_style | py::array::forcecast> buf;

    public:

      PYBIND11_TYPE_CASTER(matrix2d<data_t>, const_name("matrix2d<data_t>"));

      // Conversion part 1 (Python -> C++)
      bool load(py::handle src, bool convert)  {
        if ( !convert && !py::array_t<data_t>::check_(src) )
          return false;

        buf = py::array_t<data_t, py::array::c_style | py::array::forcecast>::ensure(src);
        if ( !buf )
          return false;

        auto dims = buf.ndim();
        if ( dims != 2  )
          return false;

        unsigned int n_rows = buf.shape()[0];
        unsigned int n_cols = buf.shape()[1];
        value.data = std::unique_ptr<data_t []>(buf.mutable_data());
        value.cols = n_cols;
        value.rows = n_rows;
        value.has_ownership = false;

        return true;
      }

      //Conversion part 2 (C++ -> Python)
      static py::handle cast(matrix2d<data_t>&& src, py::return_value_policy policy, py::handle parent)  {

        std::vector<size_t> shape {src.rows, src.cols};
        std::vector<size_t> strides { sizeof(data_t) * src.cols, sizeof(data_t) };

        py::array a(std::move(shape), std::move(strides), src.data.get() );
        src.has_ownership = false;
        
        return a.release();
      }
  };

  template<> struct type_caster<matrix1d<unsigned int>>
  {
    private:
        py::array_t<unsigned int, py::array::c_style | py::array::forcecast> buf;

    public:

      PYBIND11_TYPE_CASTER(matrix1d<unsigned int>, const_name("matrix1d<unsigned int>"));

      // Conversion part 1 (Python -> C++)
      bool load(py::handle src, bool convert)  {
        if ( !convert && !py::array_t<unsigned int>::check_(src) )
          return false;

        buf = py::array_t<unsigned int, py::array::c_style | py::array::forcecast>::ensure(src);
        if ( !buf )
          return false;

        auto dims = buf.ndim();
        if ( dims != 1  )
          return false;

        value.data = std::unique_ptr<unsigned int []>(buf.mutable_data());
        value.dim = buf.shape()[0];
        value.has_ownership = false;

        return true;
      }

      //Conversion part 2 (C++ -> Python)
      static py::handle cast(matrix1d<unsigned int>&& src, py::return_value_policy policy, py::handle parent)  {

        std::vector<size_t> shape { src.dim };
        std::vector<size_t> strides { sizeof(unsigned int) };

        py::array a(std::move(shape), std::move(strides), src.data.get() );
        src.has_ownership = false;
        
        return a.release();
      }
  };

  template<> struct type_caster<matrix1d<data_t>>
  {
    private:
        py::array_t<data_t, py::array::c_style | py::array::forcecast> buf;

    public:

      PYBIND11_TYPE_CASTER(matrix1d<data_t>, const_name("matrix1d<data_t>"));

      // Conversion part 1 (Python -> C++)
      bool load(py::handle src, bool convert)  {
        if ( !convert && !py::array_t<data_t>::check_(src) )
          return false;

        buf = py::array_t<data_t, py::array::c_style | py::array::forcecast>::ensure(src);
        if ( !buf )
          return false;

        auto dims = buf.ndim();
        if ( dims != 1  )
          return false;

        value.data = std::unique_ptr<data_t []>(buf.mutable_data());
        value.dim = buf.shape()[0];
        value.has_ownership = false;

        return true;
      }

      //Conversion part 2 (C++ -> Python)
      static py::handle cast(matrix1d<data_t>&& src, py::return_value_policy policy, py::handle parent)  {

        std::vector<size_t> shape {src.dim};
        std::vector<size_t> strides { sizeof(data_t) };

        py::array a(std::move(shape), std::move(strides), src.data.get() );
        src.has_ownership = false;
        
        return a.release();
      }
  };

}} 

template <DDT::TREE_INIT tree_init, DISTANCE::TYPES distance_type, OPTIMIZER::OPTIMIZER_TYPE tree_opt>
void bindDistanceDecisionTree(py::module& m, const std::string& suffix) {
    using TreeType = DistanceDecisionTree<tree_init, distance_type, tree_opt>;

    py::class_<TreeType>(m, ("DistanceDecisionTree" + suffix).c_str(), py::module_local())
        .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long, internal_t, internal_t>(),
             py::arg("n_classes"), py::arg("max_depth"), py::arg("max_features"),
             py::arg("seed"), py::arg("lambda"), py::arg("step_size"))
        .def("num_bytes", &TreeType::num_bytes)
        .def("num_nodes", &TreeType::num_nodes)
        .def("load", &TreeType::load, py::arg("nodes"))
        .def("predict_proba", py::overload_cast<matrix2d<data_t> const&>(&TreeType::predict_proba),
             py::arg("nodes"))
        .def("store", &TreeType::store)
        .def("fit", py::overload_cast<matrix2d<data_t> const&, matrix1d<unsigned int> const&,
                                     std::optional<std::reference_wrapper<const matrix1d<unsigned int>>>>(&TreeType::fit),
             py::arg("X"), py::arg("Y"), py::arg("indices") = py::none())
        .def("fit", py::overload_cast<matrix2d<data_t> const&, matrix1d<unsigned int> const&,
                                     matrix1d<unsigned int> const&, matrix2d<data_t> const&>(&TreeType::fit),
             py::arg("X"), py::arg("Y"), py::arg("indices"), py::arg("distance_matrix"));
}

template <DT::TREE_INIT tree_init, OPTIMIZER::OPTIMIZER_TYPE tree_opt>
void bindDecisionTree(py::module& m, const std::string& suffix) {
    using TreeType = DecisionTree<tree_init, tree_opt>;

    py::class_<TreeType>(m, ("DecisionTree" + suffix).c_str(), py::module_local())
        .def(py::init<unsigned int, unsigned int, unsigned int, unsigned long, internal_t>(),
             py::arg("n_classes"), py::arg("max_depth"), py::arg("max_features"),
             py::arg("seed"), py::arg("step_size"))
        .def("num_bytes", &TreeType::num_bytes)
        .def("num_nodes", &TreeType::num_nodes)
        .def("load", &TreeType::load, py::arg("nodes"))
        .def("predict_proba", py::overload_cast<matrix2d<data_t> const&>(&TreeType::predict_proba),
             py::arg("nodes"))
        .def("store", &TreeType::store)
        .def("fit", py::overload_cast<matrix2d<data_t> const&, matrix1d<unsigned int> const&,
                                     std::optional<std::reference_wrapper<const matrix1d<unsigned int>>>>(&TreeType::fit),
             py::arg("X"), py::arg("Y"), py::arg("indices") = py::none());
}

// Helper function to compute the size of an array at compile time
template <typename T, std::size_t N>
constexpr std::size_t n_elements(const T(&)[N]) {
    return N;
}

constexpr DDT::TREE_INIT ddt_tree_init_from_string(const char * tree_init) {
    if (strcmp(tree_init, "TRAIN") == 0 || strcmp(tree_init, "train") == 0) {
        return DDT::TREE_INIT::TRAIN;
    } else if (strcmp(tree_init, "RANDOM") == 0 || strcmp(tree_init, "random") == 0) {
        return DDT::TREE_INIT::RANDOM;
    } else {
        return DDT::TREE_INIT::TRAIN;
    }
}

constexpr DISTANCE::TYPES distance_from_string(const char * distance) {
    if (strcmp(distance, "EUCLIDEAN") == 0 || strcmp(distance, "euclidean") == 0 ) {
        return DISTANCE::TYPES::EUCLIDEAN;
    } else if (strcmp(distance, "ZLIB") == 0 || strcmp(distance, "zlib") == 0) {
        return DISTANCE::TYPES::ZLIB;
    } else if (strcmp(distance, "SHOCO") == 0 || strcmp(distance, "shoco") == 0) {
        return DISTANCE::TYPES::SHOCO;
    } else if (strcmp(distance, "LZ4") == 0 || strcmp(distance, "lz4") == 0) {
        return DISTANCE::TYPES::LZ4;
    } else {
        return DISTANCE::TYPES::LZ4;
    }
}

constexpr DT::TREE_INIT dt_tree_init_from_string(const char * tree_init) {
  if (strcmp(tree_init, "TRAIN") == 0 || strcmp(tree_init, "train") == 0) {
        return DT::TREE_INIT::TRAIN;
    } else if (strcmp(tree_init, "RANDOM") == 0 || strcmp(tree_init, "random") == 0) {
        return DT::TREE_INIT::RANDOM;
    } else {
        return DT::TREE_INIT::TRAIN;
    }
}

constexpr const char* ddt_tree_inits[] = {"TRAIN", "RANDOM"};
constexpr auto ddt_inits_max = n_elements(ddt_tree_inits);
constexpr const char* distances[] = {"ZLIB", "EUCLIDEAN", "LZ4", "SHOCO"};
constexpr auto distances_max = n_elements(distances);

template <unsigned int i, unsigned int j>
void bindDistanceDecisionTreeCombinations(py::module& m) {
    const std::string suffix = "_" + std::string(ddt_tree_inits[i]) + "_" + std::string(distances[j]);
    bindDistanceDecisionTree<ddt_tree_init_from_string(ddt_tree_inits[i]), distance_from_string(distances[j]), OPTIMIZER::NONE>(m, suffix);

    if constexpr (j < distances_max - 1) {
      bindDistanceDecisionTreeCombinations<i,j+1>(m);
    } else if constexpr (i < ddt_inits_max - 1) {
      bindDistanceDecisionTreeCombinations<i+1,0>(m);
    }
}

constexpr const char* dt_tree_inits[] = {"TRAIN", "RANDOM"};
constexpr auto dt_inits_max = n_elements(dt_tree_inits);

template <unsigned int i>
void bindDecisionTreeCombinations(py::module& m) {
    const std::string suffix = "_" + std::string(dt_tree_inits[i]);
    bindDecisionTree<dt_tree_init_from_string(ddt_tree_inits[i]), OPTIMIZER::NONE>(m, suffix);

    if constexpr (i < dt_inits_max - 1) {
      bindDecisionTreeCombinations<i+1>(m);
    }
}

PYBIND11_MODULE(ShrubEnsembles, m) {
  bindDistanceDecisionTreeCombinations<0,0>(m);
  bindDecisionTreeCombinations<0>(m);
// py::class_<OSE>(m, "COSE")
//     .def(py::init<unsigned int, unsigned int,unsigned long, bool, unsigned int, unsigned int, std::string, internal_t, std::string, std::string, std::string, internal_t>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("normalize_weights"), py::arg("burnin_steps"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("regularizer"), py::arg("l_reg"))
//     .def ("init", &OSE::init, py::arg("X"), py::arg("Y"), py::arg("n_trees"), py::arg("bootstrap"), py::arg("batch_size"))
//     .def ("next", &OSE::next, py::arg("X"), py::arg("Y"))
//     .def ("predict_proba", &OSE::predict_proba, py::arg("X"))
//     .def ("num_nodes", &OSE::num_nodes)
//     .def ("num_bytes", &OSE::num_bytes)
//     .def ("num_trees", &OSE::num_trees
// );

// py::class_<MASE>(m, "CMASE")
//     .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, unsigned int, bool, unsigned int>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("burnin_steps"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_worker"), py::arg("n_rounds"), py::arg("batch_size"), py::arg("bootstrap"), py::arg("init_tree_size"))
//     .def ("init", &MASE::init, py::arg("X"), py::arg("Y"))
//     .def ("fit", &MASE::fit, py::arg("X"), py::arg("Y"))
//     .def ("next", &MASE::next, py::arg("X"), py::arg("Y"))
//     .def ("predict_proba", &MASE::predict_proba, py::arg("X"))
//     // .def ("leafs", &PyMase::leafs)
//     // .def ("nodes", &PyMase::nodes)
//     .def ("num_nodes", &MASE::num_nodes)
//     .def ("num_bytes", &MASE::num_bytes)
//     .def ("load", &MASE::load, py::arg("new_nodes"), py::arg("new_leafs"), py::arg("new_weights"))
//     .def ("store", &MASE::store)
//     .def ("num_trees", &MASE::num_trees
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