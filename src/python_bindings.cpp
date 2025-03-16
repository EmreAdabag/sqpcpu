#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "../include/thneed.hpp"

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
        .def_readwrite("XU", &sqpcpu::Thneed::XU)
        .def_readonly("nx", &sqpcpu::Thneed::nx)
        .def_readonly("nu", &sqpcpu::Thneed::nu)
        .def_readonly("nq", &sqpcpu::Thneed::nq)
        .def_readonly("nv", &sqpcpu::Thneed::nv)
        .def_readonly("N", &sqpcpu::Thneed::N)
        .def_readonly("traj_len", &sqpcpu::Thneed::traj_len);
} 