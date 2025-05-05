#include "../include/batch_thneed.hpp"
#include <thread>

namespace sqpcpu {

BatchThneed::BatchThneed(const std::string& urdf_filename, const std::string& xml_filename, const std::string& eepos_frame_name, int batch_size, int N, 
                         float dt, int max_qp_iters, int num_threads, int fext_timesteps, float Q_cost, float dQ_cost, float R_cost, float QN_cost, float Qpos_cost, float Qvel_cost, float Qacc_cost, float orient_cost) 
    : batch_size(batch_size), N(N), dt(dt), max_qp_iters(max_qp_iters), num_threads(num_threads), fext_timesteps(fext_timesteps), Q_cost(Q_cost), 
    dQ_cost(dQ_cost), R_cost(R_cost), QN_cost(QN_cost), Qpos_cost(Qpos_cost), Qvel_cost(Qvel_cost), Qacc_cost(Qacc_cost), orient_cost(orient_cost) {
    
    // Initialize thread pool with specified number of threads or default to hardware concurrency
    int thread_count = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
    thread_pool = std::make_unique<ThreadPool>(thread_count);
    
    // Create the specified number of Thneed solvers
    solvers.reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
        solvers.emplace_back(urdf_filename, xml_filename, eepos_frame_name, N, dt, max_qp_iters, true, fext_timesteps, Q_cost, dQ_cost, R_cost, QN_cost, Qpos_cost, Qvel_cost, Qacc_cost, orient_cost);
    }

    nx = solvers[0].nx;
    nu = solvers[0].nu;
    nq = solvers[0].nq;
    nv = solvers[0].nv;
    traj_len = solvers[0].traj_len;
}

bool BatchThneed::batch_sqp(const Eigen::VectorXd& xs, 
                           const Eigen::VectorXd& eepos_g) {
    
    // Validate input sizes
    if (xs.size() != nx || eepos_g.size() != 3*N) {
        throw std::runtime_error("Input sizes do not match the number of solvers");
    }
    
    // Create a vector to store futures
    std::vector<std::future<void>> futures;
    futures.reserve(batch_size);

    // Create a vector to store per-solver update results
    std::vector<bool> updated(batch_size, true);
    
    // Submit tasks to thread pool
    for (int i = 0; i < batch_size; i++) {
        futures.push_back(
            thread_pool->enqueue(
                [this, i, &xs, &eepos_g, &updated]() {
                    updated[i] = solvers[i].sqp(xs, eepos_g);
                }
            )
        );
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
    last_state_cost = solvers[0].last_state_cost;

    // Aggregate results
    bool all_updated = true;
    for (bool u : updated) {
        all_updated &= u;
    }
    return all_updated;
}


void BatchThneed::batch_reset_solvers() {
    for (int i = 0; i < batch_size; i++) {
        solvers[i].reset_solver();
    }
}

// void BatchThneed::batch_update_xs(const Eigen::VectorXd& xs) {
//     // Validate input size
//     if (xs.size() != nx) {
//         throw std::runtime_error("Input size does not match the number of solvers");
//     }
    
//     // Create a vector to store futures
//     std::vector<std::future<void>> futures;
//     futures.reserve(batch_size);
    
//     // Submit tasks to thread pool
//     for (int i = 0; i < batch_size; i++) {
//         futures.push_back(
//             thread_pool->enqueue(
//                 [this, i, &xs]() {
//                     // solvers[i].setxs(xs);
//                 }
//             )
//         );
//     }
    
//     // Wait for all tasks to complete
//     for (auto& future : futures) {
//         future.get();
//     }
// }

void BatchThneed::batch_update_primal(const Eigen::VectorXd& XU) {
    for (int i = 0; i < batch_size; i++) {
        solvers[i].XU = XU;
    }
}

std::vector<Eigen::VectorXd> BatchThneed::get_results() const {
    std::vector<Eigen::VectorXd> results;
    results.reserve(batch_size);
    
    for (const auto& solver : solvers) {
        results.push_back(solver.XU);
    }
    
    return results;
}

std::vector<Eigen::VectorXd> BatchThneed::predict_fwd(const Eigen::VectorXd& xs, const Eigen::VectorXd& u, float dt) {
    std::vector<Eigen::VectorXd> results;
    results.reserve(batch_size);

    for (int i = 0; i < batch_size; i++) {
        // call fwd_euler and retrieve the result from xnext_tmp
        solvers[i].fwd_euler(xs, u, true, dt);
        results.push_back(solvers[i].xnext_tmp);
    }

    return results;
}


void BatchThneed::batch_set_fext(const std::vector<Eigen::MatrixXd>& fext_batch) {
    // Validate input size
    if (fext_batch.size() != batch_size) {
        throw std::runtime_error("Input batch size does not match the number of solvers");
    }
    
    // Create a vector to store futures
    std::vector<std::future<void>> futures;
    futures.reserve(batch_size);
    
    // Submit tasks to thread pool
    for (int i = 0; i < batch_size; i++) {
        futures.push_back(
            thread_pool->enqueue(
                [this, i, &fext_batch]() {
                    solvers[i].set_fext(fext_batch[i]);
                }
            )
        );
    }
    
    // Wait for all tasks to complete   
    for (auto& future : futures) {
        future.get();
    }
}

void BatchThneed::eepos(const Eigen::VectorXd& q, Eigen::Vector3d& out) {
    solvers[0].eepos(q, out);
}

void BatchThneed::eepos_and_ori(const Eigen::VectorXd& q, Eigen::Vector3d& out, Eigen::Matrix3d& ori_out) {
    solvers[0].eepos(q, out, ori_out);
}

float BatchThneed::eepos_cost(const Eigen::VectorXd& XU, const Eigen::VectorXd& eepos_g, int timesteps) {
    return solvers[0].eepos_cost(XU, eepos_g, timesteps);
}

void BatchThneed::update_goal_orientation(const Eigen::Matrix3d& goal_orientation) {
    for (int i = 0; i < batch_size; i++) {
        solvers[i].goal_orientation = goal_orientation;
    }
}

} // namespace sqpcpu 