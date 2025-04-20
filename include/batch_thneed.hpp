#ifndef SQPCPU_BATCH_THNEED_HPP
#define SQPCPU_BATCH_THNEED_HPP

#include "thneed.hpp"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>
#include <memory>

namespace sqpcpu {

class ThreadPool {
public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

class BatchThneed {
public:
    BatchThneed(const std::string& urdf_filename, const std::string& xml_filename, const std::string& eepos_frame_name, int batch_size, int N = 32, 
                float dt = 0.01, int max_qp_iters = 1, int num_threads = 0, int fext_timesteps = 0, float dQ_cost = 0.01, float R_cost = 1e-5, float QN_cost = 100.0, float Qlim_cost = 0.005, bool regularize_cost = false, float discount_factor = 0.0);
    
    int N, batch_size, num_threads, max_qp_iters, nx, nu, nq, nv, traj_len, fext_timesteps;
    float dt;

    void batch_sqp(const Eigen::VectorXd& xs, 
                  const Eigen::VectorXd& eepos_g);
    
    void batch_update_xs(const Eigen::VectorXd& xs);

    void batch_update_primal(const Eigen::VectorXd& XU);

    void batch_set_fext(const std::vector<Eigen::Vector3d>& fext_batch);

    void batch_reset_solvers();
    std::vector<Eigen::VectorXd> predict_fwd(const Eigen::VectorXd& xs, const Eigen::VectorXd& u, float dt);
    
    std::vector<Eigen::VectorXd> get_results() const;

    void eepos(const Eigen::VectorXd& q, Eigen::Vector3d& out);
    
    std::vector<Thneed> solvers;
    std::unique_ptr<ThreadPool> thread_pool;
private:
};

// ThreadPool implementation
inline ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    for(size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            while(true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { 
                        return this->stop || !this->tasks.empty(); 
                    });
                    if(this->stop && this->tasks.empty()) {
                        return;
                    }
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
    }
}

inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers) {
        worker.join();
    }
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

} // namespace sqpcpu

#endif // SQPCPU_BATCH_THNEED_HPP 