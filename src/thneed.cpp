#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include <OsqpEigen/OsqpEigen.h>
#include <iostream>
#include <iomanip>
#include <sys/time.h>

#define EEPOS_JOINT_ID 6

namespace sqpcpu {

class Thneed {
public:
    pinocchio::Model model;
    pinocchio::Data data;
    int N, nq, nv, nx, nu, nxu, traj_len;
    double dt, dQ_cost = 0.01, R_cost = 1e-5, QN_cost = 100, eps = 1.0;
    bool regularize = false;

    Eigen::MatrixXd A_k, B_k;
    Eigen::VectorXd cx_k;
    Eigen::VectorXd l;
    Eigen::VectorXd q;

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
    
    Thneed(const pinocchio::Model& model, int N=32, float dt=0.01) : model(model), data(model), N(N), dt(dt) {
        
        nq = model.nq;
        nv = model.nv;
        nx = nq + nv;
        nu = model.joints.size() - 1;
        nxu = nx + nu;
        traj_len = (nx + nu) * N - nu;

        A_k.resize(nx, nx);
        B_k.resize(nx, nu);
        cx_k.resize(nx);
        l.resize(N*nx);

        q.resize(traj_len);

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
        solver.data()->setNumberOfVariables(traj_len);
        solver.data()->setNumberOfConstraints(N*nx);
        solver.data()->setHessianMatrix(Pcsc);
        solver.data()->setGradient(q);
        solver.data()->setLinearConstraintsMatrix(Acsc);
        solver.data()->setLowerBound(l);
        solver.data()->setUpperBound(l);
        solver.initSolver();

        xnext_tmp.resize(nx);
        xcur_tmp.resize(nx);
        qpsol_tmp.resize(traj_len);
        XU_new_tmp.resize(traj_len);
        deepos_tmp.resize(3, nq);
        Q_cost_joint_err_tmp.resize(nq, nq);
    }

    void initialize_matrices() {
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

    void compute_dynamics_jacobians(const Eigen::VectorXd& q, const Eigen::VectorXd& v, const Eigen::VectorXd& u) {
        pinocchio::computeABADerivatives(model, data, q, v, u);
        
        A_k.block(nq, 0, nq, nq) = data.ddq_dq * dt;
        A_k.block(nq, nq, nq, nq) = data.ddq_dv * dt + Eigen::MatrixXd::Identity(nv, nv);
        B_k.block(nq, 0, nq, nu) = data.Minv * dt;
        
        xnext_tmp << pinocchio::integrate(model, q, v * dt), v + data.ddq * dt;
        xcur_tmp << q, v;
        
        cx_k = xnext_tmp - A_k * xcur_tmp - B_k * u;
    }

    void update_constraint_matrix(const Eigen::VectorXd& xu, const Eigen::VectorXd& xs) {
        l.segment(0, nx) = -1 * xs;
        double *Acsc_val = Acsc.valuePtr();
        int block_nnz = nx*nx + nx*nu + nx;

        for (int i = 0; i < N-1; i++) {
            int xu_stride = nx + nu;
            compute_dynamics_jacobians(
                xu.segment(i*xu_stride, nq),
                xu.segment(i*xu_stride + nq, nv),
                xu.segment(i*xu_stride + nx, nu));
            int Acsc_offset = i*block_nnz;
            for (int j = 0; j < nx; j++) {
                std::copy(A_k.data() + j*nx, A_k.data() + j*nx + nx, Acsc_val + Acsc_offset + j*nx + j + 1);
            }
            std::copy(B_k.data(), B_k.data() + nx*nu, Acsc_val + Acsc_offset + nx*nx + nx);
            l.segment((i+1)*nx, nx) = -cx_k;
        }
    }

    void fwd_euler(const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
        pinocchio::aba(model, data, x.segment(0, nq), x.segment(nq, nv), u);
        auto qnext = pinocchio::integrate(model, x.segment(0, nq), x.segment(nq, nv) * dt);
        auto vnext = x.segment(nq, nv) + data.ddq * dt;
        xnext_tmp << qnext, vnext;
    }

    void eepos(const Eigen::VectorXd& q, Eigen::Vector3d& eepos_out) {
        pinocchio::forwardKinematics(model, data, q);
        eepos_out = data.oMi[EEPOS_JOINT_ID].translation();
    }
    
    void d_eepos(const Eigen::VectorXd& q) {
        pinocchio::computeJointJacobians(model, data, q);
        deepos_tmp = pinocchio::getJointJacobian(model, data, EEPOS_JOINT_ID, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED).topRows(3);
        eepos_tmp = data.oMi[EEPOS_JOINT_ID].translation();
    }

    void update_cost_matrix(const Eigen::VectorXd& XU, const Eigen::VectorXd& eepos_g) {
        float Q_cost;
        int xu_stride = nx + nu;
        int block_nnz = nq*nq + nv + nu;
        int Pcsc_offset;
        double* Pcsc_val = Pcsc.valuePtr();
        // Eigen::VectorXd eepos_err(3);
        // Eigen::VectorXd joint_err(nq);

        for (int i = 0; i < N; i++) {
            Q_cost = i==N-1 ? QN_cost : 1.0;

            d_eepos(XU.segment(i*xu_stride, nq));
            auto joint_err = (eepos_tmp - eepos_g.segment(i*3, 3)).transpose() * deepos_tmp;
            q.segment(i*xu_stride, nq) = Q_cost * joint_err;
            q.segment(i*xu_stride + nq, nv) = dQ_cost * XU.segment(i*xu_stride + nq, nv);
            
            if (i < N-1) {
                q.segment(i*xu_stride + nx, nu) = R_cost * XU.segment(i*xu_stride + nx, nu);
            }

            Pcsc_offset = i*block_nnz;
            Q_cost_joint_err_tmp = Q_cost * joint_err.transpose() * joint_err;
            std::copy(Q_cost_joint_err_tmp.data(), Q_cost_joint_err_tmp.data() + nq*nq, Pcsc_val + Pcsc_offset);
            std::fill(Pcsc_val + Pcsc_offset + nq*nq, Pcsc_val + Pcsc_offset + nq*nq + nv, dQ_cost);
            if (i < N-1) {
                std::fill(Pcsc_val + Pcsc_offset + nq*nq + nv, Pcsc_val + Pcsc_offset + nq*nq + nv + nu, R_cost);
            }
        }
    }

    float eepos_cost(const Eigen::VectorXd& XU, const Eigen::VectorXd& eepos_g) {
        float Q_cost, cost = 0;

        for (int i = 0; i < N; i++) {
            Q_cost = i==N-1 ? QN_cost : 1.0;
            eepos(XU.segment(i*nxu, nq), eepos_tmp);
            cost += Q_cost * (eepos_tmp - eepos_g.segment(i*3, 3)).squaredNorm();
            cost += dQ_cost * XU.segment(i*nxu + nq, nv).squaredNorm();
            if (i < N-1) {
                cost += R_cost * XU.segment(i*nxu + nx, nu).squaredNorm();
            }
        }
        return cost;
    }

    float integrator_err(const Eigen::VectorXd& XU) {
        float err = 0;
        for (int i = 0; i < N-1; i++) {
            fwd_euler(XU.segment(i*nxu, nx), XU.segment(i*nxu+nx, nu));
            err += (xnext_tmp - XU.segment((i+1)*nxu, nx)).norm();
        }
        return err;
    }

    void setup_solve_osqp(Eigen::VectorXd xs, Eigen::VectorXd XU, Eigen::VectorXd eepos_g) {
        update_cost_matrix(XU, eepos_g);
        update_constraint_matrix(XU, xs);
        solver.updateHessianMatrix(Pcsc);
        solver.updateGradient(q);
        solver.updateLinearConstraintsMatrix(Acsc);
        solver.updateBounds(l, l);

        if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
            std::cout << "Error solving problem" << std::endl;
        }
        qpsol_tmp = solver.getSolution();
    }

    float linesearch(const Eigen::VectorXd& xs, const Eigen::VectorXd& XU, const Eigen::VectorXd& XU_full, const Eigen::VectorXd& eepos_g) {
        float mu = 10.0;
        float alpha = 1.0;
        float cost_new, CV_new, merit_new;

        float basecost = eepos_cost(XU, eepos_g);
        float baseCV = integrator_err(XU);
        float basemerit = basecost + mu * baseCV;

        XU_new_tmp = XU;
        for (int i = 0; i < 8; i++) {
            XU_new_tmp = XU + alpha * (XU_full - XU);
            cost_new = eepos_cost(XU_new_tmp, eepos_g);
            CV_new = integrator_err(XU_new_tmp);
            merit_new = cost_new + mu * CV_new;
            if (merit_new < basemerit) {
                return alpha;
            }
            alpha *= 0.5;
        }
        return 0;
    }

    void sqp(const Eigen::VectorXd& xs, Eigen::VectorXd& XU, const Eigen::VectorXd& eepos_g) {
        float stepsize, alpha;
        for (int i = 0; i < 2; i++) {
            setup_solve_osqp(xs, XU, eepos_g);
            
            alpha = linesearch(xs, XU, qpsol_tmp, eepos_g);

            stepsize = alpha * (qpsol_tmp - XU).norm();
            XU = XU + alpha * (qpsol_tmp - XU);
            if (stepsize < 1e-3) {
                break;
            }
        }
    }

};

} // namespace sqpcpu

int main(int argc, char ** argv)
{
  using namespace pinocchio;
  
  const std::string urdf_filename = "/home/a2rlab/Documents/emre/indy-ros2/indy_description/urdf_files/indy7.urdf";
  
  Model model;
  pinocchio::urdf::buildModel(urdf_filename,model);
  Data data(model);
  sqpcpu::Thneed t(model, 32, 0.01);

  Eigen::VectorXd q = Eigen::VectorXd::Ones(model.nq);
  Eigen::VectorXd v = Eigen::VectorXd::Ones(model.nv);
  Eigen::VectorXd u = Eigen::VectorXd::Ones(model.nv);
  Eigen::VectorXd xs = Eigen::VectorXd::Ones(model.nq + model.nv);
  Eigen::VectorXd XU = Eigen::VectorXd::Ones(t.traj_len);
  Eigen::VectorXd XU_full = Eigen::VectorXd::Zero(t.traj_len);
  Eigen::VectorXd eepos_g = Eigen::VectorXd::Ones(3*t.N);

  const int num_iterations = 1;
  struct timeval start, end;
  gettimeofday(&start, NULL);

  for (int i = 0; i < num_iterations; i++) {
    t.sqp(xs, XU, eepos_g);
  }
// t.setup_solve_osqp(xs, XU, eepos_g);
    
  gettimeofday(&end, NULL);
  double elapsed = (end.tv_sec - start.tv_sec) * 1000.0;    // sec to ms
  elapsed += (end.tv_usec - start.tv_usec) / 1000.0;        // us to ms

  std::cout << "XU: " << XU.transpose() << std::endl;
  std::cout << "Total time for " << num_iterations << " iterations: " 
            << elapsed << " ms" << std::endl;
  std::cout << "Average time per iteration: " 
            << elapsed / num_iterations << " ms\n" << std::endl;
  return 0;
}
