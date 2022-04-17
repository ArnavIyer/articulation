#include "gflags/gflags.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "amrl_msgs/AckermannCurvatureDriveMsg.h"
#include "amrl_msgs/Pose2Df.h"
#include "amrl_msgs/VisualizationMsg.h"
#include "glog/logging.h"
#include "ros/ros.h"
#include "ros/package.h"
#include "shared/math/math_util.h"
#include "shared/util/timer.h"
#include "shared/ros/ros_helpers.h"
#include "articulation.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

#include <cmath>
#include <cstdlib>

using Eigen::Vector3f;
using std::string;
using std::vector;
using std::sin;
using std::cos;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using ceres::CauchyLoss;

using namespace math_util;
using namespace ros_helpers;

namespace articulation {

Articulation::Articulation(ros::NodeHandle* n, size_t num_points) {
  num_points_ = num_points;
  point_clouds_.reserve(300);
  avg_ = {0.,0.,0.};
  optimizer_ready_ = false;
}

void Articulation::Run() {
  // This function gets called 20 times a second to form the control loop.

}

void Articulation::UpdatePointCloud(std::vector<Vector3f>& pc) {
  std::cout << "Frame Recieved: " << point_clouds_.size() << std::endl;
  std::vector<Vector3f> pcloud;
  pcloud.reserve(num_points_);

  int step = pc.size() / num_points_;
  size_t i = 0;
  if (point_clouds_.size() == 0) {
    while (pcloud.size() < num_points_) {
      // i = rand() % pc.size();
      pcloud.push_back(pc[i]);
      avg_ += pc[i];
      i += step;
    }
    avg_ /= num_points_;
  } else {
    while (pcloud.size() < num_points_) {
      // i = rand() % pc.size();
      pcloud.push_back(pc[i]);
      i += step;
    }
  }
  point_clouds_.push_back(pcloud);
  if (point_clouds_.size() == 300) {
    std::cout << "optimizer ready, avg:" << avg_ << std::endl;
    optimizer_ready_ = true;
  }
}

void Articulation::OptimizeDepthPrismatic() {
  std::cout << "starting optimize" << std::endl;

  double paa;
  double aaa;
  std::vector<double> thetas(point_clouds_.size(), 0);

  Problem problem;
  for (size_t i = 0; i < point_clouds_.size(); i++) {
    for (size_t j = 0; j < num_points_; j++) {
      CostFunction* cf = new AutoDiffCostFunction<PrismaticDepthResidual, 1, 1, 1, 1>(new PrismaticDepthResidual(
        point_clouds_[i][j](0),
        point_clouds_[i][j](1),
        point_clouds_[i][j](2),
        avg_(0), 
        avg_(1),
        avg_(2)
      ));
      problem.AddResidualBlock(cf, new CauchyLoss(0.5), &paa, &aaa, &thetas[i]);
    }
  }

  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";
  for (auto ang : thetas) {
    std::cout << ang << ",";
  }
  std::cout << std::endl;
}

void Articulation::Optimize() {
  std::cout << "starting optimize" << std::endl;

  double paa = 0;
  double aaa = 0;
  std::vector<double> thetas(point_clouds_.size(), 0);
  double offset_x = 0;
  double offset_y = 0;
  double offset_z = 0;

  for (int i = 50; i < 75; i++) {
    thetas[i] = 6.2;
  }

  Problem problem;
  for (size_t i = 0; i < point_clouds_.size(); i++) {
    for (size_t j = 0; j < num_points_; j++) {
      CostFunction* cf = new AutoDiffCostFunction<RevoluteDepthResidual, 1, 1, 1, 1,1,1,1>(new RevoluteDepthResidual(
        point_clouds_[i][j](0),
        point_clouds_[i][j](1),
        point_clouds_[i][j](2),
        avg_(0),
        avg_(1),
        avg_(2)
      ));
      problem.AddResidualBlock(cf, new CauchyLoss(0.5), &paa, &aaa, &thetas[i], &offset_x,&offset_y,&offset_z);
      
    }
  }
  
  problem.SetParameterLowerBound(&paa, 0, 0.0);
  problem.SetParameterUpperBound(&paa, 0, M_PI * 2);
  problem.SetParameterLowerBound(&aaa, 0, 0.0);
  problem.SetParameterUpperBound(&aaa, 0, M_PI * 2);
  for (size_t i = 0; i < thetas.size(); i++) {
    problem.SetParameterLowerBound(&thetas[i], 0, 0.0);
    problem.SetParameterUpperBound(&thetas[i], 0, M_PI * 2);
  }
  problem.SetParameterUpperBound(&offset_x, 0, 10.);
  problem.SetParameterLowerBound(&offset_x, 0, -10.);
  problem.SetParameterUpperBound(&offset_y, 0, 10.);
  problem.SetParameterLowerBound(&offset_y, 0, -10.);
  problem.SetParameterUpperBound(&offset_z, 0, 10.);
  problem.SetParameterLowerBound(&offset_z, 0, -10.);

  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";
  std::cout << offset_x << std::endl;
  for (auto ang : thetas) {
    std::cout << ang << ",";
  }
  std::cout << std::endl;
}

}  // namespace navigation
