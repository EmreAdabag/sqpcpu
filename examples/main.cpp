#include "../include/thneed.hpp"
#include <iostream>
#include <iomanip>
#include <sys/time.h>

int main(int argc, char ** argv)
{
  using namespace pinocchio;
  
  const std::string urdf_filename = "/Users/emreadabag/code/indy-ros2/indy_description/urdf_files/indy7.urdf";
  
  sqpcpu::Thneed t2(urdf_filename, 4, 0.01, 5);
  t2.XU = Eigen::VectorXd::Ones(t2.traj_len);

  Eigen::VectorXd xs = Eigen::VectorXd::Ones(t2.nx);
  Eigen::VectorXd eepos_g = Eigen::VectorXd::Ones(3*t2.N);

  const int num_iterations = 1;
  struct timeval start, end;
  gettimeofday(&start, NULL);

  for (int i = 0; i < num_iterations; i++) {
    t2.sqp(xs, eepos_g);
  }
    
  gettimeofday(&end, NULL);
  double elapsed = (end.tv_sec - start.tv_sec) * 1000.0;    // sec to ms
  elapsed += (end.tv_usec - start.tv_usec) / 1000.0;        // us to ms

  std::cout << "XU2: " << t2.XU.transpose() << std::endl;
  std::cout << "Total time for " << num_iterations << " iterations: " 
            << elapsed << " ms" << std::endl;
  std::cout << "Average time per iteration: " 
            << elapsed / num_iterations << " ms\n" << std::endl;
  return 0;
} 