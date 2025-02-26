#include "sqpcpu/thneed.hpp"
#include <iostream>
#include <iomanip>
#include <sys/time.h>

int main(int argc, char ** argv)
{
  using namespace pinocchio;
  
  const std::string urdf_filename = "/home/a2rlab/Documents/emre/indy-ros2/indy_description/urdf_files/indy7.urdf";
  
  Model model;
  pinocchio::urdf::buildModel(urdf_filename,model);
  Data data(model);
  sqpcpu::Thneed t(model, 4);

  Eigen::VectorXd xs = Eigen::VectorXd::Ones(model.nq + model.nv);
  Eigen::VectorXd XU = Eigen::VectorXd::Ones(t.get_traj_len());
  Eigen::VectorXd eepos_g = Eigen::VectorXd::Ones(3*t.get_N());

  const int num_iterations = 1;
  struct timeval start, end;
  gettimeofday(&start, NULL);

  for (int i = 0; i < num_iterations; i++) {
    t.sqp(xs, XU, eepos_g);
  }
    
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