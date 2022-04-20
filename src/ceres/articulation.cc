#include "articulation.h"
#include "amrl_msgs/AckermannCurvatureDriveMsg.h"
#include "amrl_msgs/Pose2Df.h"
#include "amrl_msgs/VisualizationMsg.h"
#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "ros/package.h"
#include "ros/ros.h"
#include "shared/math/math_util.h"
#include "shared/ros/ros_helpers.h"
#include "shared/util/timer.h"

#include <cmath>
#include <cstdlib>

using Eigen::Vector3d;
using std::cos;
using std::sin;
using std::string;
using std::vector;

using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

using namespace math_util;
using namespace ros_helpers;

namespace articulation {

Articulation::Articulation(ros::NodeHandle *n, size_t num_points) {
  num_points_ = num_points;
  point_clouds_.reserve(300);
  avg_ = {0., 0., 0.};
}

void Articulation::UpdatePointCloud(std::vector<Vector3d> &pc) {
  std::cout << "Frame Recieved: " << point_clouds_.size() << std::endl;
  std::vector<Vector3d> pcloud;
  pcloud.reserve(num_points_);

  int step = pc.size() / num_points_;
  size_t i = 0;
  if (point_clouds_.size() == 0) {
    while (pcloud.size() < num_points_) {
      pcloud.push_back(pc[i]);
      avg_ += pc[i];
      i += step;
    }
    avg_ /= num_points_;
  } else {
    while (pcloud.size() < num_points_) {
      pcloud.push_back(pc[i]);
      i += step;
    }
  }
  point_clouds_.push_back(pcloud);
}

// --------------------- ONLINE OPTIMIZATION FUNCTIONS -------------------------

void Articulation::OptimizePrismaticOnline() {

}

void Articulation::OptimizeRevoluteOnline() {}

// --------------------- OFFLINE OPTIMIZATION FUNCTIONS ------------------------

void Articulation::OptimizePrismaticOffline() {
  double paa = 0;
  double aaa = 0;
  std::vector<double> thetas(point_clouds_.size(), 0);

  Problem problem;
  for (size_t i = 0; i < point_clouds_.size(); i++) {
    for (size_t j = 0; j < num_points_; j++) {
      CostFunction *cf =
          new AutoDiffCostFunction<PrismaticDepthResidual, 1, 1, 1, 1>(
              new PrismaticDepthResidual(point_clouds_[i][j], avg_));
      problem.AddResidualBlock(cf, new CauchyLoss(0.5), &paa, &aaa, &thetas[i]);
    }
  }

  problem.SetParameterLowerBound(&paa, 0, -1 * M_PI * 2);
  problem.SetParameterUpperBound(&paa, 0, M_PI * 2);
  problem.SetParameterLowerBound(&aaa, 0, -1 * M_PI * 2);
  problem.SetParameterUpperBound(&aaa, 0, M_PI * 2);

  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";
  std::cout << "normal_vector: {" << sin(paa) * cos(aaa) << ","
            << sin(paa) * sin(aaa) << "," << cos(paa) << "}" << std::endl;
  std::cout << "thetas: " << std::flush;
  for (auto ang : thetas) {
    std::cout << ang << ",";
  }
  std::cout << std::endl;
}

void Articulation::OptimizeRevoluteOffline() {
  std::cout << "starting optimize" << std::endl;

  double paa = 0;
  double aaa = 0;
  std::vector<double> thetas(point_clouds_.size(), 0);
  double offset_x = 0;
  double offset_y = 0;
  double offset_z = 0;

  Problem problem;
  for (size_t i = 0; i < point_clouds_.size(); i++) {
    for (size_t j = 0; j < num_points_; j++) {
      CostFunction *cf =
          new AutoDiffCostFunction<RevoluteDepthResidual, 1, 1, 1, 1, 1, 1, 1>(
              new RevoluteDepthResidual(point_clouds_[i][j], avg_));
      problem.AddResidualBlock(cf, new CauchyLoss(0.5), &paa, &aaa, &thetas[i],
                               &offset_x, &offset_y, &offset_z);
    }
  }

  problem.SetParameterLowerBound(&paa, 0, -1 * M_PI * 2);
  problem.SetParameterUpperBound(&paa, 0, M_PI * 2);
  problem.SetParameterLowerBound(&aaa, 0, -1 * M_PI * 2);
  problem.SetParameterUpperBound(&aaa, 0, M_PI * 2);
  for (size_t i = 0; i < thetas.size(); i++) {
    problem.SetParameterLowerBound(&thetas[i], 0, -1 * M_PI * 2);
    problem.SetParameterUpperBound(&thetas[i], 0, M_PI * 2);
  }

  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";
  std::cout << "normal_vector: {" << sin(paa) * cos(aaa) << ","
            << sin(paa) * sin(aaa) << "," << cos(paa) << "}" << std::endl;
  std::cout << "thetas: " << std::flush;
  for (auto ang : thetas) {
    std::cout << ang << ",";
  }
  std::cout << std::endl;
}

} // namespace articulation
