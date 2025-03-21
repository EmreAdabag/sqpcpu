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
    BatchThneed(const std::string& urdf_filename, int batch_size, int N = 32, 
                float dt = 0.01, int max_qp_iters = 1, int num_threads = 0);
    
    void batch_sqp(const std::vector<Eigen::VectorXd>& xs_batch, 
                  const std::vector<Eigen::VectorXd>& eepos_g_batch);
    
    void batch_update_xs(const std::vector<Eigen::VectorXd>& xs_batch);
    
    std::vector<Eigen::VectorXd> get_results() const;

    std::vector<Thneed> solvers;
    std::unique_ptr<ThreadPool> thread_pool;
    int batch_size;
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