#ifndef SQPCPU_THNEED_HPP
#define SQPCPU_THNEED_HPP

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>

#define EEPOS_JOINT_ID 6

namespace sqpcpu {

class Thneed {
public:

    Thneed(const std::string& urdf_filename, const int N=32, const float dt=0.01, const int max_qp_iters=1, const bool osqp_warm_start=true);
    
    void initialize_matrices();
    void compute_dynamics_jacobians(const Eigen::VectorXd& q, const Eigen::VectorXd& v, const Eigen::VectorXd& u);
    void update_constraint_matrix(const Eigen::VectorXd& xs);
    void setxs(const Eigen::VectorXd& xs);
    void fwd_euler(const Eigen::VectorXd& x, const Eigen::VectorXd& u);
    void eepos(const Eigen::VectorXd& q, Eigen::Vector3d& eepos_out);
    void d_eepos(const Eigen::VectorXd& q);
    void update_cost_matrix(const Eigen::VectorXd& eepos_g);
    float eepos_cost(const Eigen::VectorXd& XU, const Eigen::VectorXd& eepos_g);
    float integrator_err(const Eigen::VectorXd& XU);
    bool setup_solve_osqp(Eigen::VectorXd xs, Eigen::VectorXd eepos_g);
    float linesearch(const Eigen::VectorXd& xs, const Eigen::VectorXd& XU_full, const Eigen::VectorXd& eepos_g);
    void sqp(const Eigen::VectorXd& xs, const Eigen::VectorXd& eepos_g);

    pinocchio::Model model;
    pinocchio::Data data;
    int N, nq, nv, nx, nu, nxu, traj_len, max_qp_iters;
    double dt, dQ_cost = 0.01, R_cost = 1e-5, QN_cost = 100, eps = 1.0;
    bool regularize = false;
    bool osqp_warm_start;

    Eigen::MatrixXd A_k, B_k;
    Eigen::VectorXd cx_k;
    Eigen::VectorXd l;
    Eigen::VectorXd q;
    Eigen::VectorXd XU;

    Eigen::SparseMatrix<double> Acsc;
    Eigen::SparseMatrix<double> Pcsc;

    OsqpEigen::Solver solver;

    // temporary variables
    Eigen::VectorXd q_tmp;
    Eigen::VectorXd v_tmp;
    Eigen::VectorXd u_tmp;
    Eigen::VectorXd xs_tmp;
    Eigen::VectorXd XU_tmp;
    Eigen::VectorXd qpsol_tmp;
    Eigen::Vector3d eepos_tmp;
    Eigen::MatrixXd Q_cost_joint_err_tmp;
    Eigen::VectorXd XU_new_tmp;
    Eigen::MatrixXd deepos_tmp;
    Eigen::VectorXd xnext_tmp;
    Eigen::VectorXd xcur_tmp;
private:
};

} // namespace sqpcpu

#endif // SQPCPU_THNEED_HPP 