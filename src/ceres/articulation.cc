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

Articulation::Articulation(ros::NodeHandle *n, const size_t& num_iters, const std::string& type) : avg_(0,0,0), type_(type) {
  thetas_.reserve(300);
  times_.reserve(300);
  options_.linear_solver_type = ceres::DENSE_QR;
  options_.max_num_iterations = num_iters;
  options_.minimizer_progress_to_stdout = false;
}

void Articulation::Print() {
  if (type_ == "prismatic")
    std::cout << "normal_vector: {" << sin(paa_) * cos(aaa_) << ", "
              << sin(paa_) * sin(aaa_) << ", " << cos(paa_) << "}" << std::endl;
  else if (type_ == "revolute") {
    Eigen::Vector3d n = {n_x_, n_y_, n_z_};
    n /= n.norm();
    std::cout << "normal_vector: {" << n.x() << "," << n.y() << "," << n.z() << "}" << std::endl;
  }
  std::cout << "offset vector: {" << offset_x_ << ", " << offset_y_ << ", " << offset_z_ << "}" << std::endl;
  std::cout << "thetas: " << std::flush;
  for (auto ang : thetas_) {
    std::cout << ang << ",";
  }
  std::cout << std::endl << "times: ";
  for (auto a : times_) {
    std::cout << a << ",";
  }
  std::cout << std::endl;
}

void Articulation::UpdatePointCloud(std::vector<Vector3d> &pc, const size_t& n) {
  point_cloud_.clear();

  size_t step = pc.size() / (n + 1);
  size_t i = 0;
  if (first_update) {
    first_update = false;
    while (point_cloud_.size() < n) {
      point_cloud_.push_back(pc.at(i));
      avg_ += pc.at(i);
      i += step;
    }
    avg_ /= n;
  } else {
    while (point_cloud_.size() < n) {
      point_cloud_.push_back(pc[i]);
      i += step;
    }
  }
}

// --------------------- ONLINE OPTIMIZATION FUNCTIONS -------------------------

void Articulation::OptimizePrismaticOnline(const double& time) {
  if (point_cloud_.empty()) {
    return;
  }

  thetas_.push_back(0);
  times_.push_back(time);

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

void Articulation::OptimizeRevoluteOnline(const double& time) {
  if (point_cloud_.empty()) {
    return;
  }

  thetas_.push_back(0);
  times_.push_back(time);

  for (size_t j = 0; j < point_cloud_.size(); j++) {
    CostFunction *cf =
        new AutoDiffCostFunction<RevoluteDepthResidual3Norm, 1, 1, 1, 1, 1, 1, 1, 1>(
            new RevoluteDepthResidual3Norm(point_cloud_[j], avg_));
    problem_.AddResidualBlock(cf, new CauchyLoss(0.5), &n_x_, &n_y_, &n_z_, &thetas_.back(), &offset_x_, &offset_y_, &offset_z_);
  }

  if (!added_bounds_) {
    problem_.SetParameterLowerBound(&n_x_, 0, -2);
    problem_.SetParameterUpperBound(&n_x_, 0, 2);
    problem_.SetParameterLowerBound(&n_y_, 0, -2);
    problem_.SetParameterUpperBound(&n_y_, 0, 2);
    problem_.SetParameterLowerBound(&n_z_, 0, -2);
    problem_.SetParameterUpperBound(&n_z_, 0, 2);
    added_bounds_ = true;
  }

  problem_.SetParameterLowerBound(&thetas_.back(), 0, -1 * M_PI);
  problem_.SetParameterUpperBound(&thetas_.back(), 0, M_PI);

  Solve(options_, &problem_, &summary_);

  point_cloud_.clear();
}
} // namespace articulation
