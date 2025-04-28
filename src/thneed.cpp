#include "../include/thneed.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/mjcf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/fwd.hpp"
#include <OsqpEigen/OsqpEigen.h>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cmath>
#define VERBOSE 0


namespace sqpcpu {

    Thneed::Thneed(const std::string& urdf_filename, const std::string& xml_filename, const std::string& eepos_frame_name, int N, float dt, const int max_qp_iters, const bool osqp_warm_start, const int fext_timesteps, float Q_cost, float dQ_cost, float R_cost, float QN_cost, float Qlim_cost_unused, float orient_cost) : 
        N(N), dt(dt), max_qp_iters(max_qp_iters), osqp_warm_start(osqp_warm_start), fext_timesteps(fext_timesteps), Q_cost(Q_cost), dQ_cost(dQ_cost), R_cost(R_cost), QN_cost(QN_cost), Qlim_cost_unused(Qlim_cost_unused), orient_cost(orient_cost) {

        if (urdf_filename.empty()) {
            pinocchio::mjcf::buildModel(xml_filename, model);
        } else {
            pinocchio::urdf::buildModel(urdf_filename, model);
        }
        data = pinocchio::Data(model);
        joint_limits_lower = model.lowerPositionLimit;
        joint_limits_upper = model.upperPositionLimit;

        eepos_joint_id = 6; // just used for setting external forces
        eepos_frame_id = model.getFrameId(eepos_frame_name);
        if (VERBOSE) {
            std::cout << "eepos_frame_id: " << eepos_frame_id << std::endl;
            std::cout << "joint limits lower size: " << joint_limits_lower.size() << std::endl;
            std::cout << "joint limits upper size: " << joint_limits_upper.size() << std::endl;
            std::cout << "joint limits lower: " << joint_limits_lower.transpose() << std::endl;
            std::cout << "joint limits upper: " << joint_limits_upper.transpose() << std::endl;
        }
        
        nq = model.nq;
        nv = model.nv;
        nx = nq + nv;
        nu = model.njoints - 1; // unclear where this comes from
        nxu = nx + nu;
        traj_len = (nx + nu) * N - nu;
        if (VERBOSE) {
            std::cout << "nq, nv, nu: " << nq << ", " << nv << ", " << nu << std::endl;
        }

        A_k.resize(nx, nx);
        B_k.resize(nx, nu);
        cx_k.resize(nx);
        l.resize(N*nx);
        XU.resize(traj_len); XU.setZero();
        q.resize(traj_len);
        xnext_tmp.resize(nx);
        xcur_tmp.resize(nx);
        qpsol_tmp.resize(traj_len);
        XU_new_tmp.resize(traj_len);
        deepos_tmp.resize(3, nq);
        deepos_ori_tmp.resize(3, nq);
        eepos_tmp.resize(3);
        eepos_ori_tmp.resize(3, 3);
        dpose_tmp.resize(6, nq);
        Q_cost_joint_err_tmp.resize(nq, nq);
        fext = pinocchio::container::aligned_vector<pinocchio::Force>(model.njoints, pinocchio::Force::Zero());

        goal_orientation.resize(3, 3);
        goal_orientation << 0.707107, 0, -0.707107,
                            0, 1, 0,
                            0.707107, 0, 0.707107;

        // top left corner of A_k is I
        A_k.topLeftCorner(nq, nq) = Eigen::MatrixXd::Identity(nq, nq);
        // top right corner of A_k is dt*I
        A_k.block(0, nq, nq, nq) = dt * Eigen::MatrixXd::Identity(nq, nq);
        // set B to zeros
        B_k.setZero();

        Acsc.resize(N*nx, traj_len);
        Pcsc.resize(traj_len, traj_len);
        initialize_matrices();

        solver.settings()->setVerbosity(false);
        solver.settings()->setWarmStart(osqp_warm_start);
        solver.data()->setNumberOfVariables(traj_len);
        solver.data()->setNumberOfConstraints(N*nx);
        solver.data()->setHessianMatrix(Pcsc);
        solver.data()->setGradient(q);
        solver.data()->setLinearConstraintsMatrix(Acsc);
        solver.data()->setLowerBound(l);
        solver.data()->setUpperBound(l);
        solver.initSolver();
    }

    void Thneed::reset_solver() {
        solver.clearSolverVariables();
        XU.setZero();
        fext = pinocchio::container::aligned_vector<pinocchio::Force>(model.njoints, pinocchio::Force::Zero());
    }

    void Thneed::initialize_matrices() {
        std::vector<Eigen::Triplet<double>> P_triplets, A_triplets;
        P_triplets.reserve(N*nq*nq + N*nv + (N-1)*nu);
        A_triplets.reserve(N*nx*nx + (N-1)*(nx*(nx+nu)));

        // initialize Pcsc
        for (int block = 0; block < N; block++) {
            for (int i = 0; i < nq; i++) {
                for (int j = 0; j < nq; j++) {
                    P_triplets.push_back(Eigen::Triplet<double>(block*nxu + i, block*nxu + j, 1.0));
                }
            }
            for (int i = 0; i < nv; i++) {
                P_triplets.push_back(Eigen::Triplet<double>(block*nxu + nq + i, block*nxu + nq + i, 1.0));
            }
            if (block < N-1) {
                for (int i = 0; i < nu; i++) {
                    P_triplets.push_back(Eigen::Triplet<double>(block*nxu + nx + i, block*nxu + nx + i, 1.0));
                }
            }
        }

        for (int block = 0; block < N; block++) {
            if (block > 0) {
                for (int i = 0; i < nx; i++) {
                    for (int j = 0; j < nx; j++) {
                        A_triplets.push_back(Eigen::Triplet<double>(block*nx + i, (block-1)*nxu + j, 1.0));
                    }
                }
                for (int i = 0; i < nx; i++) {
                    for (int j = 0; j < nu; j++) {
                        A_triplets.push_back(Eigen::Triplet<double>(block*nx + i, (block-1)*nxu + nx + j, 1.0));
                    }
                }
            }
            for (int i = 0; i < nx; i++) {
                A_triplets.push_back(Eigen::Triplet<double>(block*nx + i, block*nxu + i, -1.0));
            }
        }

        Pcsc.setFromTriplets(P_triplets.begin(), P_triplets.end());
        Acsc.setFromTriplets(A_triplets.begin(), A_triplets.end());
    }

    // void Thneed::setxs(const Eigen::VectorXd& xs) {
    //     XU.segment(0, nx) = xs;
    // }

    void Thneed::compute_dynamics_jacobians(const Eigen::VectorXd& q, const Eigen::VectorXd& v, const Eigen::VectorXd& u, bool usefext) {
        if (usefext) {
            pinocchio::computeABADerivatives(model, data, q, v, u, fext);
        } else {
            pinocchio::computeABADerivatives(model, data, q, v, u);
        }
        
        A_k.block(nq, 0, nq, nq) = data.ddq_dq * dt;
        A_k.block(nq, nq, nq, nq) = data.ddq_dv * dt + Eigen::MatrixXd::Identity(nv, nv);
        B_k.block(nq, 0, nq, nu) = data.Minv * dt;
        
        xnext_tmp << pinocchio::integrate(model, q, v * dt), v + data.ddq * dt;
        xcur_tmp << q, v;
        
        cx_k = xnext_tmp - A_k * xcur_tmp - B_k * u;
    }

    void Thneed::update_constraint_matrix(const Eigen::VectorXd& xs) {
        l.segment(0, nx) = -1 * xs;
        double *Acsc_val = Acsc.valuePtr();
        int block_nnz = nx*nx + nx*nu + nx;

        for (int i = 0; i < N-1; i++) {
            int xu_stride = nx + nu;
            bool usefext = i < fext_timesteps;
            // std::cout << "usefext: " << usefext << " fext_timesteps: " << fext_timesteps << std::endl;
            compute_dynamics_jacobians(
                XU.segment(i*xu_stride, nq),
                XU.segment(i*xu_stride + nq, nv),
                XU.segment(i*xu_stride + nx, nu),
                usefext);
            int Acsc_offset = i*block_nnz;
            for (int j = 0; j < nx; j++) {
                std::copy(A_k.data() + j*nx, A_k.data() + j*nx + nx, Acsc_val + Acsc_offset + j*nx + j + 1);
            }
            std::copy(B_k.data(), B_k.data() + nx*nu, Acsc_val + Acsc_offset + nx*nx + nx);
            l.segment((i+1)*nx, nx) = -cx_k;
        }
    }

    void Thneed::fwd_euler(const Eigen::VectorXd& x, const Eigen::VectorXd& u, bool usefext, float dt) {
        if (dt == 0.0) { dt = this->dt; }
        if (usefext) {
            pinocchio::aba(model, data, x.segment(0, nq), x.segment(nq, nv), u, fext);
        } else {
            pinocchio::aba(model, data, x.segment(0, nq), x.segment(nq, nv), u);
        }
        
        auto qnext = pinocchio::integrate(model, x.segment(0, nq), x.segment(nq, nv) * dt);
        auto vnext = x.segment(nq, nv) + data.ddq * dt;
        xnext_tmp << qnext, vnext;
    }

    void Thneed::eepos(const Eigen::VectorXd& q, Eigen::Vector3d& eepos_out) {
        pinocchio::forwardKinematics(model, data, q);
        eepos_out = pinocchio::updateFramePlacement(model, data, eepos_frame_id).translation();
    }

    void Thneed::eepos(const Eigen::VectorXd& q, Eigen::Vector3d& eepos_out, Eigen::Matrix3d& eepos_ori_out) {
        pinocchio::forwardKinematics(model, data, q);
        eepos_out = pinocchio::updateFramePlacement(model, data, eepos_frame_id).translation();
        eepos_ori_out = pinocchio::updateFramePlacement(model, data, eepos_frame_id).rotation();
    }
    
    void Thneed::d_eepos(const Eigen::VectorXd& q) {
        // pinocchio::computeJointJacobians(model, data, q);
        pinocchio::computeFrameJacobian(model, data, q, eepos_frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, dpose_tmp);
        deepos_tmp = dpose_tmp.topRows(3);
        deepos_ori_tmp = dpose_tmp.block(3, 0, 3, nq);
        eepos_tmp = data.oMf[eepos_frame_id].translation();
        eepos_ori_tmp = data.oMf[eepos_frame_id].rotation();
    }

    Eigen::MatrixXd Thneed::compute_rotation_error(const Eigen::Matrix3d& R_current, const Eigen::Matrix3d& R_desired) {
        // Error rotation matrix: R_e = R_d * R_c^T
        Eigen::Matrix3d R_error = R_desired * R_current.transpose();
        // Convert to axis-angle representation using Eigen
        Eigen::AngleAxisd angle_axis(R_error);
        return -1.0 * angle_axis.angle() * angle_axis.axis();
    }
    

    void Thneed::update_cost_matrix(const Eigen::VectorXd& eepos_g) {
        float Q_cost_i;
        int xu_stride = nx + nu;
        int block_nnz = nq*nq + nv + nu;
        int Pcsc_offset;
        double* Pcsc_val = Pcsc.valuePtr();

        Eigen::MatrixXd joint_err(nq, 1);
        Eigen::MatrixXd joint_ori_err(nq, 1);
        Eigen::MatrixXd dist_min(nq, 1);
        Eigen::MatrixXd dist_max(nq, 1);
        Eigen::VectorXd joint_limit_jac(nq);

        for (int i = 0; i < N; i++) {
            Q_cost_i = i==N-1 ? QN_cost : Q_cost;
            
            d_eepos(XU.segment(i*xu_stride, nq));
            
            joint_err = (eepos_tmp - eepos_g.segment(i*3, 3)).transpose() * deepos_tmp;
            joint_ori_err = compute_rotation_error(eepos_ori_tmp, goal_orientation).transpose() * deepos_ori_tmp;
            // dist_min = XU.segment(i*xu_stride, nq) - joint_limits_lower;
            // dist_max = joint_limits_upper - XU.segment(i*xu_stride, nq);
            // joint_limit_jac = -dist_min.cwiseInverse() + dist_max.cwiseInverse();
            q.segment(i*xu_stride, nq) = Q_cost_i * joint_err + orient_cost * joint_ori_err; // + Qlim_cost * joint_limit_jac;
            q.segment(i*xu_stride + nq, nv) = dQ_cost * XU.segment(i*xu_stride + nq, nv);
            
            if (i < N-1) {
                q.segment(i*xu_stride + nx, nu) = R_cost * XU.segment(i*xu_stride + nx, nu);
            }

            Pcsc_offset = i*block_nnz;
            Q_cost_joint_err_tmp = Q_cost_i * joint_err.transpose() * joint_err + orient_cost * joint_ori_err * joint_ori_err.transpose();
            std::copy(Q_cost_joint_err_tmp.data(), Q_cost_joint_err_tmp.data() + nq*nq, Pcsc_val + Pcsc_offset);
            std::fill(Pcsc_val + Pcsc_offset + nq*nq, Pcsc_val + Pcsc_offset + nq*nq + nv, dQ_cost);
            if (i < N-1) {
                std::fill(Pcsc_val + Pcsc_offset + nq*nq + nv, Pcsc_val + Pcsc_offset + nq*nq + nv + nu, R_cost);
            }
        }
    }

    float Thneed::eepos_cost(const Eigen::VectorXd& xu, const Eigen::VectorXd& eepos_g, int timesteps) {
        // computes cost for timesteps timesteps
        if (timesteps == -1) { timesteps = N; }
        float Q_cost_i, cost = 0;
        float dist2, stage_cost;

        for (int i = 0; i < timesteps; i++) {
            Q_cost_i = i==N-1 ? QN_cost : Q_cost;

            stage_cost = 0.0;
            eepos(xu.segment(i*nxu, nq), eepos_tmp, eepos_ori_tmp);
            dist2 = (eepos_tmp - eepos_g.segment(i*3, 3)).squaredNorm();
            
            stage_cost += Q_cost_i * dist2; // quadratic cost
            stage_cost += orient_cost * compute_rotation_error(eepos_ori_tmp, goal_orientation).squaredNorm();
            stage_cost += dQ_cost * xu.segment(i*nxu + nq, nv).squaredNorm();
            // stage_cost += -Qlim_cost * log((xu.segment(i*nxu, nq) - joint_limits_lower).array()).sum();
            // stage_cost += -Qlim_cost * log((joint_limits_upper - xu.segment(i*nxu, nq)).array()).sum();
            if (i < timesteps-1) {
                stage_cost += R_cost * xu.segment(i*nxu + nx, nu).squaredNorm();
            }
            cost += stage_cost;
        }
        return cost;
    }

    float Thneed::integrator_err(const Eigen::VectorXd& xu) {
        float err = 0;
        for (int i = 0; i < N-1; i++) {
            bool usefext = i < fext_timesteps;
            fwd_euler(xu.segment(i*nxu, nx), xu.segment(i*nxu+nx, nu), usefext);
            err += (xnext_tmp - xu.segment((i+1)*nxu, nx)).norm();
        }
        return err;
    }

    bool Thneed::setup_solve_osqp(Eigen::VectorXd xs, Eigen::VectorXd eepos_g) {
        update_cost_matrix(eepos_g);
        update_constraint_matrix(xs);
        solver.updateHessianMatrix(Pcsc);
        solver.updateGradient(q);
        solver.updateLinearConstraintsMatrix(Acsc);
        solver.updateBounds(l, l);

        auto errflag = solver.solveProblem();
        if (VERBOSE) {
            std::cout << "Solver status: " << static_cast<int>(solver.getStatus()) << std::endl;
        }
        if (errflag != OsqpEigen::ErrorExitFlag::NoError) {
            // print solver status in a way that doesn't require stream operator
            std::cout << "Solver error: " << static_cast<int>(errflag) << std::endl;
            std::cout << "Solver status: " << static_cast<int>(solver.getStatus()) << std::endl;
            return false;
        }
        qpsol_tmp = solver.getSolution();
        return true;
    }

    float Thneed::linesearch(const Eigen::VectorXd& xs, const Eigen::VectorXd& XU_full, const Eigen::VectorXd& eepos_g) {
        float mu = 10.0;
        float alpha = 1.0;
        float cost_new, CV_new, merit_new;

        float basecost = eepos_cost(XU, eepos_g);
        float baseCV = integrator_err(XU) + (XU.segment(0, nx) - xs).norm();
        float basemerit = basecost + mu * baseCV;

        if (VERBOSE) {
            std::cout << "basecost: " << basecost << ", baseCV: " << baseCV << ", basemerit: " << basemerit << std::endl;
        }

        // XU_new_tmp = XU;
        for (int i = 0; i < 8; i++) {
            XU_new_tmp = XU + alpha * (XU_full - XU);
            cost_new = eepos_cost(XU_new_tmp, eepos_g);
            CV_new = integrator_err(XU_new_tmp) + (XU_new_tmp.segment(0, nx) - xs).norm();
            merit_new = cost_new + mu * CV_new;
            if (VERBOSE) {
                std::cout << "cost_new: " << cost_new << ", CV_new: " << CV_new << ", merit_new: " << merit_new << std::endl;
            }
            if (merit_new < basemerit) {
                return alpha;
            }
            alpha *= 0.5;
        }
        return 0;
    }

    void Thneed::sqp(const Eigen::VectorXd& xs, const Eigen::VectorXd& eepos_g) {

        if (VERBOSE) {
            std::cout << "xs: " << xs.transpose() << std::endl;
            std::cout << "eepos_g: " << eepos_g.transpose() << std::endl;
            std::cout << "XU: " << XU.transpose() << std::endl;
        }

        XU.segment(0, nx) = xs;
        last_state_cost = eepos_cost(xs, eepos_g, 1); // for stats

        float stepsize, alpha;
        for (int i = 0; i < max_qp_iters; i++) {
            if (!setup_solve_osqp(xs, eepos_g)) { continue; }
            
            alpha = linesearch(xs, qpsol_tmp, eepos_g);
            if (alpha == 0.0) { continue; }

            stepsize = alpha * (qpsol_tmp - XU).norm();
            XU = XU + alpha * (qpsol_tmp - XU);
            if (stepsize < 1e-3) {
                break;
            }
        }
    }

    void Thneed::set_fext(const Eigen::MatrixXd& f_ext) {
        // Set external force/torque at each joint in world frame
        // Create force object with both linear force and torque
        // Assume f_ext is 6*njoints dimensional vector with [fx,fy,fz,tx,ty,tz] for each joint
        for (size_t i = 0; i < model.njoints; ++i) {
            Eigen::Vector3d linear = f_ext.block<3, 1>(0, i);
            Eigen::Vector3d torque = f_ext.block<3, 1>(3, i);
            pinocchio::Force force(linear, torque);
            fext[i] = force;
        }
    }

} // namespace sqpcpu