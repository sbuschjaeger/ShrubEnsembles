
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Datatypes.h"
#include "Matrix.h"

#include "GASE.h"
#include "MASE.h"
#include "OSE.h"
#include "DecisionTree.h"

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
      static py::handle cast(const matrix2d<data_t>& src, py::return_value_policy policy, py::handle parent)  {

        std::vector<size_t> shape {src.rows, src.cols};
        std::vector<size_t> strides { sizeof(data_t) * src.cols, sizeof(data_t) };

        py::array a(std::move(shape), std::move(strides), src.data.get() );
        //src.has_ownership = false;
        
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
      static py::handle cast(const matrix1d<unsigned int>& src, py::return_value_policy policy, py::handle parent)  {

        std::vector<size_t> shape {src.dim};
        std::vector<size_t> strides { sizeof(unsigned int) * src.dim, sizeof(unsigned int) };

        py::array a(std::move(shape), std::move(strides), src.data.get() );
        //src.has_ownership = false;
        
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
      static py::handle cast(const matrix1d<data_t>& src, py::return_value_policy policy, py::handle parent)  {

        std::vector<size_t> shape {src.dim};
        std::vector<size_t> strides { sizeof(data_t) * src.dim, sizeof(data_t) };

        py::array a(std::move(shape), std::move(strides), src.data.get() );
        //src.has_ownership = false;
        
        return a.release();

      }
  };

}} 


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
    .def(py::init<unsigned int, unsigned int,unsigned long, unsigned int, unsigned int, std::string, internal_t, std::string, std::string, unsigned int, unsigned int, unsigned int, unsigned int, bool>(), py::arg("n_classes"), py::arg("max_depth"), py::arg("seed"), py::arg("burnin_steps"), py::arg("max_features"), py::arg("loss"), py::arg("step_size"), py::arg("optimizer"),  py::arg("tree_init_mode"), py::arg("n_trees"), py::arg("n_worker"), py::arg("n_rounds"), py::arg("batch_size"), py::arg("bootstrap"))
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
);

}