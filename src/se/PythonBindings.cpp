
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Datatypes.h"
#include "Matrix.h"

// #include "DecisionTree.h"
// #include "GASE.h"
// #include "MASE.h"
// #include "OSE.h"
#include "DecisionTree.h"

// struct inty { long long_value; };

// inty sum(inty & s) {
//     s.long_value += 5;
//     return s;
//     // std::cout << s.long_value << std::endl;
// }

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
        // std::make_unique<data_t[]>(buf.mutable_data());
        //value._data = std::make_unique<data_t[]>(std::move(buf.mutable_data()));
        value.cols = n_cols;
        value.rows = n_rows;
        value.has_ownership = false;
        // value = std::move(matrix2d<data_t>(n_rows, n_cols, buf.mutable_data(), false)); //, buf.data(), buf.data()+buf.size());
        //value._data = buf.mutable_data();
        //std::cout << n_rows << " " << n_cols << std::endl;
        //std::cout << value(0,0) << " " << value(0,1) << std::endl;
        //std::copy(buf.data(), buf.data() + n_rows*n_cols, value.begin());

        return true;
      }

      //Conversion part 2 (C++ -> Python)
      static py::handle cast(const matrix2d<data_t>& src, py::return_value_policy policy, py::handle parent)  {

        std::vector<size_t> shape {src.rows, src.cols};
        std::vector<size_t> strides { sizeof(data_t) * src.cols, sizeof(data_t) };

        //std::vector<size_t> strides(3);

        // for ( int i = 0 ; i < 3 ; ++i ) {
        //   shape  [i] = src.shape  [i];
        //   strides[i] = src.strides[i]*sizeof(T);
        // }

        py::array a(std::move(shape), std::move(strides), src.data.get() );
        //src.has_ownership = false;
        
        return a.release();

      }
  };
}} // namespace pybind11::detail

// namespace pybind11 { namespace detail {
//     template <> struct type_caster<matrix<data_t>> {
//     public:
//         /**
//          * This macro establishes the name 'inty' in
//          * function signatures and declares a local variable
//          * 'value' of type inty
//          */
//         PYBIND11_TYPE_CASTER(matrix<data_t>, const_name("matrix<data_t>"));

//         /**
//          * Conversion part 1 (Python->C++): convert a PyObject into a inty
//          * instance or return false upon failure. The second argument
//          * indicates whether implicit conversions should be applied.
//          */
//         bool load(handle src, bool) {
//             if ( !convert and !py::array_t<data_t>::check_(src) )
//                 return false;

//             auto buf = py::array_t<data_t, py::array::c_style | py::array::forcecast>::ensure(src);
//             if ( !buf )
//                 return false;

//             auto dims = buf.ndim();
//             if ( dims != 2  )
//                 return false;

//             unsigned int n_rows = buf.shape()[0];
//             unsigned int n_cols = buf.shape()[1];

//             value = matrix<data_t>(n_rows, n_cols); //, buf.data(), buf.data()+buf.size());
            
//             std::copy(buf.data(), buf.data() + n_rows*n_cols, value.begin());
//         }

//         /**
//          * Conversion part 2 (C++ -> Python): convert an inty instance into
//          * a Python object. The second and third arguments are used to
//          * indicate the return value policy and parent object (for
//          * ``return_value_policy::reference_internal``) and are generally
//          * ignored by implicit casters.
//          */
//         static handle cast(matrix<data_t> src, return_value_policy /* policy */, handle /* parent */) {
//             // return PyLong_FromLong(1.0);
//             std::vector<size_t> shape {src.rows(), src.cols()};
//             std::vector<size_t> strides { sizeof(data_t) * src.cols(), sizeof(data_t) };

//             // std::vector<size_t> strides(3);

//             // for ( int i = 0 ; i < 3 ; ++i ) {
//             //   shape  [i] = src.shape  [i];
//             //   strides[i] = src.strides[i]*sizeof(T);
//             // }

//             py::array a(std::move(shape), std::move(strides), src.data() );

//             return a.release();
//         }
//     };
// }} // namespace pybind11::detail



// data_t sum(matrix<data_t> &matrix) {
//     data_t s = 0;
//     for (unsigned int i = 0; i < matrix.rows(); ++i) {
//         for (unsigned int j = 0; j < matrix.cols(); ++j) {
//             s += matrix(i,j);
//         }
//     }
//     return s;
// }

PYBIND11_MODULE(CShrubEnsembles, m) {

// m.def("sum", &sum, "A function that adds two numbers");

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
//     .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, unsigned int, bool>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("burnin_steps"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_worker"), py::arg("n_rounds"), py::arg("batch_size"), py::arg("bootstrap"))
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
//     // .def ("weights", &PyMase::weights)
//     // .def ("update_weights", &PyMase::update_weights, py::arg("new_weights"))
//     // .def ("update_nodes", &PyMase::update_nodes, py::arg("new_nodes"))
//     // .def ("update_leafs", &PyMase::update_leafs, py::arg("update_leafs"))
//     .def ("num_trees", &MASE::num_trees
// );

// py::class_<GASE>(m, "CGASE")
//     .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, unsigned int, bool>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_batchs"), py::arg("n_rounds"),py::arg("init_batch_size"), py::arg("bootstrap"))
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
//     // .def ("update_weights", &PyGase::update_weights, py::arg("new_weights"))
//     // .def ("update_nodes", &PyGase::update_nodes, py::arg("new_nodes"))
//     // .def ("update_nodes", &PyGase::update_nodes, py::arg("new_nodes"))
//     // .def ("update_leafs", &PyGase::update_leafs, py::arg("update_leafs"))
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
//     // .def ("update_leafs", &DecisionTreeClassifier::update_leafs, py::arg("new_leafs"))
//     // .def ("leafs", &DecisionTreeClassifier::leafs)
//     // .def ("update_nodes", &DecisionTreeClassifier::update_nodes, py::arg("new_nodes"))
//     // .def ("nodes", &PyDecisionTreeClassifier::nodes
// );

// py::class_<matrix<data_t>>(m, "matrix", py::buffer_protocol())
//    .def_buffer([](matrix<data_t> &m) -> py::buffer_info {
//         return py::buffer_info(
//             m.data(),                               /* Pointer to buffer */
//             sizeof(data_t),                          /* Size of one scalar */
//             py::format_descriptor<data_t>::format(), /* Python struct-style format descriptor */
//             2,                                      /* Number of dimensions */
//             { m.rows(), m.cols() },                 /* Buffer dimensions */
//             { sizeof(data_t) * m.cols(),             /* Strides (in bytes) for each index */
//               sizeof(data_t) }
//         );
//     }
// );

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