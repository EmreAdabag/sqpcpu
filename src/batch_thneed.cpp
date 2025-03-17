#include "../include/batch_thneed.hpp"
#include <thread>

namespace sqpcpu {

BatchThneed::BatchThneed(const std::string& urdf_filename, int batch_size, int N, 
                         float dt, int max_qp_iters, int num_threads) 
    : batch_size(batch_size) {
    
    // Initialize thread pool with specified number of threads or default to hardware concurrency
    int thread_count = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
    thread_pool = std::make_unique<ThreadPool>(thread_count);
    
    // Create the specified number of Thneed solvers
    solvers.reserve(batch_size);
    for (int i = 0; i < batch_size; i++) {
        solvers.emplace_back(urdf_filename, N, dt, max_qp_iters);
    }
}

void BatchThneed::batch_sqp(const std::vector<Eigen::VectorXd>& xs_batch, 
                           const std::vector<Eigen::VectorXd>& eepos_g_batch) {
    
    // Validate input sizes
    if (xs_batch.size() != batch_size || eepos_g_batch.size() != batch_size) {
        throw std::runtime_error("Input batch sizes do not match the number of solvers");
    }
    
    // Create a vector to store futures
    std::vector<std::future<void>> futures;
    futures.reserve(batch_size);
    
    // Submit tasks to thread pool
    for (int i = 0; i < batch_size; i++) {
        futures.push_back(
            thread_pool->enqueue(
                [this, i, &xs_batch, &eepos_g_batch]() {
                    solvers[i].sqp(xs_batch[i], eepos_g_batch[i]);
                }
            )
        );
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
}

void BatchThneed::batch_update_xs(const std::vector<Eigen::VectorXd>& xs_batch) {
    // Validate input size
    if (xs_batch.size() != batch_size) {
        throw std::runtime_error("Input batch size does not match the number of solvers");
    }
    
    // Create a vector to store futures
    std::vector<std::future<void>> futures;
    futures.reserve(batch_size);
    
    // Submit tasks to thread pool
    for (int i = 0; i < batch_size; i++) {
        futures.push_back(
            thread_pool->enqueue(
                [this, i, &xs_batch]() {
                    solvers[i].setxs(xs_batch[i]);
                }
            )
        );
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
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

} // namespace sqpcpu 