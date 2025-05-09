#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "../include/thneed.hpp"
#include "../include/batch_thneed.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pysqpcpu, m) {
    m.doc() = "Python bindings for the Thneed SQP solver";
    
    py::class_<sqpcpu::Thneed>(m, "Thneed")
        .def(py::init<const std::string&, const std::string&, const std::string&, const int, const float, const int, const bool, const int, const float, const float, const float, const float, const float, const float, const float, const float>(),
             py::arg("urdf_filename") = "",
             py::arg("xml_filename") = "",
             py::arg("eepos_frame_name") = "end_effector",
             py::arg("N") = 32,
             py::arg("dt") = 0.01,
             py::arg("max_qp_iters") = 1,
             py::arg("osqp_warm_start") = true,
             py::arg("fext_timesteps") = 0,
             py::arg("Q_cost") = 1.0,
             py::arg("dQ_cost") = 0.01,
             py::arg("R_cost") = 1e-5,
             py::arg("QN_cost") = 100.0,
             py::arg("Qpos_cost") = 0.0,
             py::arg("Qvel_cost") = 0.0,
             py::arg("Qacc_cost") = 0.0,
             py::arg("orient_cost") = 0.0)
        .def("sqp", &sqpcpu::Thneed::sqp,
             py::arg("xs"),
             py::arg("eepos_g"))
        // .def("setxs", &sqpcpu::Thneed::setxs,
        //      py::arg("xs"))
        .def("eepos", [](sqpcpu::Thneed& self, const Eigen::VectorXd& q) {
            Eigen::Vector3d eepos_out;
            self.eepos(q, eepos_out);
            return eepos_out;
        },
             py::arg("q"))
        .def("eepos_cost", &sqpcpu::Thneed::eepos_cost,
             py::arg("XU"),
             py::arg("eepos_g"),
             py::arg("timesteps") = -1)
        .def("set_fext", &sqpcpu::Thneed::set_fext,
             py::arg("f_ext"))
        .def("reset_solver", &sqpcpu::Thneed::reset_solver)
        .def_readwrite("XU", &sqpcpu::Thneed::XU)
        .def_readonly("nx", &sqpcpu::Thneed::nx)
        .def_readonly("nu", &sqpcpu::Thneed::nu)
        .def_readonly("nq", &sqpcpu::Thneed::nq)
        .def_readonly("nv", &sqpcpu::Thneed::nv)
        .def_readonly("N", &sqpcpu::Thneed::N)
        .def_readonly("traj_len", &sqpcpu::Thneed::traj_len)
        .def_readonly("last_state_cost", &sqpcpu::Thneed::last_state_cost)
        .def_readwrite("goal_orientation", &sqpcpu::Thneed::goal_orientation);

    py::class_<sqpcpu::BatchThneed>(m, "BatchThneed")
        .def(py::init<const std::string&, const std::string&, const std::string&, int, int, float, int, int, int, float, float, float, float, float, float, float, float>(),
             py::arg("urdf_filename") = "",
             py::arg("xml_filename") = "",
             py::arg("eepos_frame_name") = "end_effector",
             py::arg("batch_size"),
             py::arg("N") = 32,
             py::arg("dt") = 0.01,
             py::arg("max_qp_iters") = 1,
             py::arg("num_threads") = 0,
             py::arg("fext_timesteps") = 0,
             py::arg("Q_cost") = 1.0,
             py::arg("dQ_cost") = 0.01,
             py::arg("R_cost") = 1e-5,
             py::arg("QN_cost") = 100.0,
             py::arg("Qpos_cost") = 0.0,
             py::arg("Qvel_cost") = 0.0,
             py::arg("Qacc_cost") = 0.0,
             py::arg("orient_cost") = 0.0)
        .def("sqp", &sqpcpu::BatchThneed::batch_sqp,
             py::arg("xs"),
             py::arg("eepos_g"))
     //    .def("batch_update_xs", &sqpcpu::BatchThneed::batch_update_xs,
     //         py::arg("xs"))
        .def("batch_update_primal", &sqpcpu::BatchThneed::batch_update_primal,
             py::arg("XU"))
        .def("batch_set_fext", &sqpcpu::BatchThneed::batch_set_fext,
             py::arg("fext_batch"))
        .def("get_results", &sqpcpu::BatchThneed::get_results)
        .def("reset_solver", &sqpcpu::BatchThneed::batch_reset_solvers)
        .def("eepos", [](sqpcpu::BatchThneed& self, const Eigen::VectorXd& q) {
            Eigen::Vector3d eepos_out;
            self.eepos(q, eepos_out);
            return eepos_out;
        },
             py::arg("q"))
        .def("eepos_and_ori", [](sqpcpu::BatchThneed& self, const Eigen::VectorXd& q) {
            Eigen::Vector3d eepos_out;
            Eigen::Matrix3d eepos_ori_out;
            self.eepos_and_ori(q, eepos_out, eepos_ori_out);
            return std::make_tuple(eepos_out, eepos_ori_out);
        },
             py::arg("q"))
        .def("eepos_cost", &sqpcpu::BatchThneed::eepos_cost,
             py::arg("XU"),
             py::arg("eepos_g"),
             py::arg("timesteps") = -1)
        .def("predict_fwd", &sqpcpu::BatchThneed::predict_fwd,
             py::arg("xs"),
             py::arg("u"),
             py::arg("dt"))
        .def("update_goal_orientation", &sqpcpu::BatchThneed::update_goal_orientation,
             py::arg("goal_orientation"))
        .def_readonly("N", &sqpcpu::BatchThneed::N)
        .def_readonly("nx", &sqpcpu::BatchThneed::nx)
        .def_readonly("nu", &sqpcpu::BatchThneed::nu)
        .def_readonly("nq", &sqpcpu::BatchThneed::nq)
        .def_readonly("nv", &sqpcpu::BatchThneed::nv)
        .def_readonly("traj_len", &sqpcpu::BatchThneed::traj_len)
        .def_readonly("fext_timesteps", &sqpcpu::BatchThneed::fext_timesteps)
        .def_readonly("last_state_cost", &sqpcpu::BatchThneed::last_state_cost);
} 