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

Articulation::Articulation(ros::NodeHandle *n) : avg_(0,0,0) {
  thetas_.reserve(300);
  options_.linear_solver_type = ceres::DENSE_QR;
  options_.max_num_iterations = 10;
  options_.minimizer_progress_to_stdout = true;
}

void Articulation::Print() {
  std::cout << "normal_vector: {" << sin(paa_) * cos(aaa_) << ", "
            << sin(paa_) * sin(aaa_) << ", " << cos(paa_) << "}" << std::endl;
  std::cout << "offset vector: {" << offset_x_ << ", " << offset_y_ << ", " << offset_z_ << "}" << std::endl;
  std::cout << "thetas: " << std::flush;
  for (auto ang : thetas_) {
    std::cout << ang << ",";
  }
}

void Articulation::UpdatePointCloud(std::vector<Vector3d> &pc) {
  point_cloud_ = std::move(pc);
}

// --------------------- ONLINE OPTIMIZATION FUNCTIONS -------------------------

void Articulation::OptimizePrismaticOnline() {
  if (point_cloud_.empty()) {
    return;
  }

  std::cout << "inside optimize" << std::endl;

  thetas_.push_back(0);

  for (size_t j = 0; j < point_cloud_.size(); j++) {
    CostFunction *cf =
        new AutoDiffCostFunction<PrismaticDepthResidual, 1, 1, 1, 1>(
            new PrismaticDepthResidual(point_cloud_[j], avg_));
    problem_.AddResidualBlock(cf, new CauchyLoss(0.5), &paa_, &aaa_, &thetas_.back());
  }

  if (!added_bounds_) {
    problem_.SetParameterLowerBound(&paa_, 0, -1 * M_PI * 2);
    problem_.SetParameterUpperBound(&paa_, 0, M_PI * 2);
    problem_.SetParameterLowerBound(&aaa_, 0, -1 * M_PI * 2);
    problem_.SetParameterUpperBound(&aaa_, 0, M_PI * 2);
    added_bounds_ = true;
  }

  Solve(options_, &problem_, &summary_);

  point_cloud_.clear();
}

void Articulation::OptimizeRevoluteOnline() {}

// --------------------- OFFLINE OPTIMIZATION FUNCTIONS ------------------------

// void Articulation::OptimizePrismaticOffline() {
//   double paa = 0;
//   double aaa = 0;
//   std::vector<double> thetas(point_clouds_.size(), 0);

//   Problem problem;
//   for (size_t i = 0; i < point_clouds_.size(); i++) {
//     for (size_t j = 0; j < num_points_; j++) {
//       CostFunction *cf =
//           new AutoDiffCostFunction<PrismaticDepthResidual, 1, 1, 1, 1>(
//               new PrismaticDepthResidual(point_clouds_[i][j], avg_));
//       problem.AddResidualBlock(cf, new CauchyLoss(0.5), &paa, &aaa, &thetas[i]);
//     }
//   }

//   problem.SetParameterLowerBound(&paa, 0, -1 * M_PI * 2);
//   problem.SetParameterUpperBound(&paa, 0, M_PI * 2);
//   problem.SetParameterLowerBound(&aaa, 0, -1 * M_PI * 2);
//   problem.SetParameterUpperBound(&aaa, 0, M_PI * 2);

//   Solver::Options options;
//   options.linear_solver_type = ceres::DENSE_QR;
//   options.max_num_iterations = 100;
//   options.minimizer_progress_to_stdout = true;

//   Solver::Summary summary;
//   Solve(options, &problem, &summary);

//   std::cout << summary.FullReport() << "\n";
//   std::cout << "normal_vector: {" << sin(paa) * cos(aaa) << ","
//             << sin(paa) * sin(aaa) << "," << cos(paa) << "}" << std::endl;
//   std::cout << "thetas: " << std::flush;
//   for (auto ang : thetas) {
//     std::cout << ang << ",";
//   }
//   std::cout << std::endl;
// }

// void Articulation::OptimizeRevoluteOffline() {
//   std::cout << "starting optimize" << std::endl;

//   double paa = 0;
//   double aaa = 0;
//   std::vector<double> thetas(point_clouds_.size(), 0);
//   double offset_x = 0;
//   double offset_y = 0;
//   double offset_z = 0;

//   Problem problem;
//   for (size_t i = 0; i < point_clouds_.size(); i++) {
//     for (size_t j = 0; j < num_points_; j++) {
//       CostFunction *cf =
//           new AutoDiffCostFunction<RevoluteDepthResidual, 1, 1, 1, 1, 1, 1, 1>(
//               new RevoluteDepthResidual(point_clouds_[i][j], avg_));
//       problem.AddResidualBlock(cf, new CauchyLoss(0.5), &paa, &aaa, &thetas[i],
//                                &offset_x, &offset_y, &offset_z);
//     }
//   }

//   problem.SetParameterLowerBound(&paa, 0, -1 * M_PI * 2);
//   problem.SetParameterUpperBound(&paa, 0, M_PI * 2);
//   problem.SetParameterLowerBound(&aaa, 0, -1 * M_PI * 2);
//   problem.SetParameterUpperBound(&aaa, 0, M_PI * 2);
//   for (size_t i = 0; i < thetas.size(); i++) {
//     problem.SetParameterLowerBound(&thetas[i], 0, -1 * M_PI * 2);
//     problem.SetParameterUpperBound(&thetas[i], 0, M_PI * 2);
//   }

//   Solver::Options options;
//   options.linear_solver_type = ceres::DENSE_QR;
//   options.max_num_iterations = 100;
//   options.minimizer_progress_to_stdout = true;

//   Solver::Summary summary;
//   Solve(options, &problem, &summary);

//   std::cout << summary.FullReport() << "\n";
//   std::cout << "normal_vector: {" << sin(paa) * cos(aaa) << ","
//             << sin(paa) * sin(aaa) << "," << cos(paa) << "}" << std::endl;
//   std::cout << "thetas: " << std::flush;
//   for (auto ang : thetas) {
//     std::cout << ang << ",";
//   }
//   std::cout << std::endl;
// }

} // namespace articulation
