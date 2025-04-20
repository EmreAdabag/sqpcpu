#include "../include/batch_thneed.hpp"
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <vector>

int main(int argc, char ** argv)
{
  using namespace pinocchio;
  
  const std::string urdf_filename = "../urdfs/indy7.urdf";
  
  // Parameters
  const int batch_size = 4;
  const int N = 32;
  const float dt = 0.01;
  const int max_qp_iters = 5;
  const int num_threads = 4;  // Use 4 threads
  
  // Create a BatchThneed instance
  sqpcpu::BatchThneed batch_solver(urdf_filename, "", "end_effector", batch_size, N, dt, max_qp_iters, num_threads);
  
  // Create batch inputs
  std::vector<Eigen::VectorXd> xs_batch;
  std::vector<Eigen::VectorXd> eepos_g_batch;
  
  // Get the dimensions from the first solver
  int nx = 12;  // This should match the nx from Thneed
  
  // Initialize batch inputs with different values
  Eigen::VectorXd xs = Eigen::VectorXd::Zero(nx);
  Eigen::VectorXd eepos_g = Eigen::VectorXd::Ones(3 * N);
  
  
  // Measure execution time
  struct timeval start, end;
  gettimeofday(&start, NULL);
  
  // Run batch SQP
  batch_solver.batch_sqp(xs, eepos_g);
  
  gettimeofday(&end, NULL);
  double elapsed = (end.tv_sec - start.tv_sec) * 1000.0;    // sec to ms
  elapsed += (end.tv_usec - start.tv_usec) / 1000.0;        // us to ms
  
  // Get results
  std::vector<Eigen::VectorXd> results = batch_solver.get_results();
  
  // Print results
  for (int i = 0; i < batch_size; i++) {
    std::cout << "Result " << i << ": " << results[i].transpose().head(10) << " ..." << std::endl;
  }
  
  std::cout << "Total execution time: " << elapsed << " ms" << std::endl;
  std::cout << "Average time per problem: " << elapsed / batch_size << " ms" << std::endl;
  
  return 0;
} 