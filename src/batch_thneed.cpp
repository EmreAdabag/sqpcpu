#include "../include/batch_thneed.hpp"
#include <thread>

namespace sqpcpu {

BatchThneed::BatchThneed(const std::string& urdf_filename, const std::string& xml_filename, const std::string& eepos_frame_name, int batch_size, int N, 
                         float dt, int max_qp_iters, int num_threads, int fext_timesteps, float dQ_cost, float R_cost, float QN_cost, float Qlim_cost, bool regularize_cost, float discount_factor) 
    : batch_size(batch_size), N(N), dt(dt), max_qp_iters(max_qp_iters), num_threads(num_threads), fext_timesteps(fext_timesteps) {
    
    // Initialize thread pool with specified number of threads or default to hardware concurrency
    int thread_count = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
    thread_pool = std::make_unique<ThreadPool>(thread_count);
    
    // Create the specified number of Thneed solvers
    solvers.reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
        solvers.emplace_back(urdf_filename, xml_filename, eepos_frame_name, N, dt, max_qp_iters, true, fext_timesteps, dQ_cost, R_cost, QN_cost, Qlim_cost, regularize_cost, discount_factor);
    }

    nx = solvers[0].nx;
    nu = solvers[0].nu;
    nq = solvers[0].nq;
    nv = solvers[0].nv;
    traj_len = solvers[0].traj_len;
}

void BatchThneed::batch_sqp(const Eigen::VectorXd& xs, 
                           const Eigen::VectorXd& eepos_g) {
    
    // Validate input sizes
    if (xs.size() != nx || eepos_g.size() != 3*N) {
        throw std::runtime_error("Input sizes do not match the number of solvers");
    }
    
    // Create a vector to store futures
    std::vector<std::future<void>> futures;
    futures.reserve(batch_size);
    
    // Submit tasks to thread pool
    for (int i = 0; i < batch_size; i++) {
        futures.push_back(
            thread_pool->enqueue(
                [this, i, &xs, &eepos_g]() {
                    solvers[i].sqp(xs, eepos_g);
                }
            )
        );
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
}

void BatchThneed::batch_reset_solvers() {
    for (int i = 0; i < batch_size; i++) {
        solvers[i].reset_solver();
    }
}

void BatchThneed::batch_update_xs(const Eigen::VectorXd& xs) {
    // Validate input size
    if (xs.size() != nx) {
        throw std::runtime_error("Input size does not match the number of solvers");
    }
    
    // Create a vector to store futures
    std::vector<std::future<void>> futures;
    futures.reserve(batch_size);
    
    // Submit tasks to thread pool
    for (int i = 0; i < batch_size; i++) {
        futures.push_back(
            thread_pool->enqueue(
                [this, i, &xs]() {
                    solvers[i].setxs(xs);
                }
            )
        );
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
}

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


void BatchThneed::batch_set_fext(const std::vector<Eigen::Vector3d>& fext_batch) {
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

} // namespace sqpcpu 