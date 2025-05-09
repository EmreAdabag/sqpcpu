#ifndef SQPCPU_THNEED_HPP
#define SQPCPU_THNEED_HPP

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>

#define EEPOS_JOINT_ID 6

namespace sqpcpu {

class Thneed {
public:

    Thneed(const std::string& urdf_filename="", const std::string& xml_filename="", const std::string& eepos_frame_name="end_effector", 
           const int N=32, const float dt=0.01, const int max_qp_iters=1, const bool osqp_warm_start=true, const int fext_timesteps=0, 
           float Q_cost=1.0, float dQ_cost=0.01, float R_cost=1e-5, float QN_cost=10, float Qpos_cost=0.0, float Qvel_cost=0.0, float Qacc_cost=0.0, float orient_cost=0.0);
    
    void initialize_matrices();
    void reset_solver();
    void compute_dynamics_jacobians(const Eigen::VectorXd& q, const Eigen::VectorXd& v, const Eigen::VectorXd& u, bool usefext=false);
    void update_constraint_matrix(const Eigen::VectorXd& xs);
    void fwd_euler(const Eigen::VectorXd& x, const Eigen::VectorXd& u, bool usefext=false, float dt=0.0);
    void eepos(const Eigen::VectorXd& q, Eigen::Vector3d& eepos_out);
    void eepos(const Eigen::VectorXd& q, Eigen::Vector3d& eepos_out, Eigen::Matrix3d& eepos_ori_out);
    Eigen::MatrixXd compute_rotation_error(const Eigen::Matrix3d& R_current, const Eigen::Matrix3d& R_desired);
    void d_eepos(const Eigen::VectorXd& q);
    void update_cost_matrix(const Eigen::VectorXd& eepos_g);
    float eepos_cost(const Eigen::VectorXd& XU, const Eigen::VectorXd& eepos_g, int timesteps=-1);
    float integrator_err(const Eigen::VectorXd& XU);
    bool setup_solve_osqp(Eigen::VectorXd xs, Eigen::VectorXd eepos_g);
    float linesearch(const Eigen::VectorXd& xs, const Eigen::VectorXd& XU_full, const Eigen::VectorXd& eepos_g);
    bool sqp(const Eigen::VectorXd& xs, const Eigen::VectorXd& eepos_g);
    void set_fext(const Eigen::MatrixXd& f_ext);

    pinocchio::Model model;
    pinocchio::Data data;
    Eigen::VectorXd joint_limits_lower, joint_limits_upper;
    float joint_limit_margin;
    int eepos_joint_id, eepos_frame_id;
    float dt, Q_cost, dQ_cost, R_cost, QN_cost, Qpos_cost, Qvel_cost, Qacc_cost, orient_cost, last_state_cost;
    int N, nq, nv, nx, nu, nxu, traj_len, max_qp_iters, fext_timesteps;
    bool osqp_warm_start;
    pinocchio::container::aligned_vector<pinocchio::Force> fext;

    Eigen::MatrixXd A_k, B_k;
    Eigen::VectorXd cx_k;
    Eigen::VectorXd l;
    Eigen::VectorXd q;
    Eigen::VectorXd XU;

    Eigen::SparseMatrix<double> Acsc;
    Eigen::SparseMatrix<double> Pcsc;

    OsqpEigen::Solver solver;

    // temporary variables
    Eigen::MatrixXd goal_orientation;
    Eigen::VectorXd q_tmp;
    Eigen::VectorXd v_tmp;
    Eigen::VectorXd u_tmp;
    Eigen::VectorXd xs_tmp;
    Eigen::VectorXd XU_tmp;
    Eigen::VectorXd qpsol_tmp;
    Eigen::Vector3d eepos_tmp;
    Eigen::Matrix3d eepos_ori_tmp;
    Eigen::MatrixXd deepos_ori_tmp;
    Eigen::MatrixXd Q_cost_joint_err_tmp;
    Eigen::VectorXd XU_new_tmp;
    Eigen::MatrixXd deepos_tmp;
    Eigen::MatrixXd dpose_tmp;
    Eigen::VectorXd xnext_tmp;
    Eigen::VectorXd xcur_tmp;
private:
};

} // namespace sqpcpu

#endif // SQPCPU_THNEED_HPP 