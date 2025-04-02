#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "../include/thneed.hpp"
#include "../include/batch_thneed.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pysqpcpu, m) {
    m.doc() = "Python bindings for the Thneed SQP solver";
    
    py::class_<sqpcpu::Thneed>(m, "Thneed")
        .def(py::init<const std::string&, const int, const float, const int>(),
             py::arg("urdf_filename"),
             py::arg("N") = 32,
             py::arg("dt") = 0.01,
             py::arg("max_qp_iters") = 1)
        .def("sqp", &sqpcpu::Thneed::sqp,
             py::arg("xs"),
             py::arg("eepos_g"))
        .def("setxs", &sqpcpu::Thneed::setxs,
             py::arg("xs"))
        .def("eepos", [](sqpcpu::Thneed& self, const Eigen::VectorXd& q) {
            Eigen::Vector3d eepos_out;
            self.eepos(q, eepos_out);
            return eepos_out;
        },
             py::arg("q"))
        .def("set_fext", &sqpcpu::Thneed::set_fext,
             py::arg("f_ext"))
        .def_readwrite("XU", &sqpcpu::Thneed::XU)
        .def_readonly("nx", &sqpcpu::Thneed::nx)
        .def_readonly("nu", &sqpcpu::Thneed::nu)
        .def_readonly("nq", &sqpcpu::Thneed::nq)
        .def_readonly("nv", &sqpcpu::Thneed::nv)
        .def_readonly("N", &sqpcpu::Thneed::N)
        .def_readonly("traj_len", &sqpcpu::Thneed::traj_len);

    py::class_<sqpcpu::BatchThneed>(m, "BatchThneed")
        .def(py::init<const std::string&, int, int, float, int, int, int, float, float, float>(),
             py::arg("urdf_filename"),
             py::arg("batch_size"),
             py::arg("N") = 32,
             py::arg("dt") = 0.01,
             py::arg("max_qp_iters") = 1,
             py::arg("num_threads") = 0,
             py::arg("fext_timesteps") = 0,
             py::arg("dQ_cost") = 0.01,
             py::arg("R_cost") = 1e-5,
             py::arg("QN_cost") = 100.0)
        .def("batch_sqp", &sqpcpu::BatchThneed::batch_sqp,
             py::arg("xs_batch"),
             py::arg("eepos_g_batch"))
        .def("batch_update_xs", &sqpcpu::BatchThneed::batch_update_xs,
             py::arg("xs_batch"))
        .def("batch_set_fext", &sqpcpu::BatchThneed::batch_set_fext,
             py::arg("fext_batch"))
        .def("get_results", &sqpcpu::BatchThneed::get_results)
        .def("eepos", [](sqpcpu::BatchThneed& self, const Eigen::VectorXd& q) {
            Eigen::Vector3d eepos_out;
            self.eepos(q, eepos_out);
            return eepos_out;
        },
             py::arg("q"))
        .def("predict_fwd", &sqpcpu::BatchThneed::predict_fwd,
             py::arg("xs"),
             py::arg("u"),
             py::arg("dt"))
        .def_readonly("N", &sqpcpu::BatchThneed::N)
        .def_readonly("nx", &sqpcpu::BatchThneed::nx)
        .def_readonly("nu", &sqpcpu::BatchThneed::nu)
        .def_readonly("nq", &sqpcpu::BatchThneed::nq)
        .def_readonly("nv", &sqpcpu::BatchThneed::nv)
        .def_readonly("traj_len", &sqpcpu::BatchThneed::traj_len)
        .def_readonly("fext_timesteps", &sqpcpu::BatchThneed::fext_timesteps);
} 