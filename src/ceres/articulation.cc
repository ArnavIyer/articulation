#include "articulation.h"
#include "amrl_msgs/AckermannCurvatureDriveMsg.h"
#include "amrl_msgs/Pose2Df.h"
#include "amrl_msgs/VisualizationMsg.h"
#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "geometry_msgs/Point.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "ros/package.h"
#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
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
namespace rvt = rviz_visual_tools;

namespace articulation {

Articulation::Articulation(ros::NodeHandle *n, const size_t &num_iters,
                           const std::string &type,
                           ros::Publisher *cloud_publisher)
    : axis_(0, 0, 1), offset_(0.0, 0.0, 0.0), type_(type) {
  thetas_.reserve(300);
  times_.reserve(300);
  options_.linear_solver_type = ceres::DENSE_QR;
  options_.max_num_iterations = num_iters;
  options_.minimizer_progress_to_stdout = false;
  visual_tools_.reset(
      new rviz_visual_tools::RvizVisualTools("base_link", "/optimization_viz"));

  cloud_publisher_ = cloud_publisher;

  cloud_msg_.header.seq = 0;
  cloud_msg_.header.frame_id = "base_link";
  cloud_msg_.fields.resize(4);
  cloud_msg_.point_step = 3 * sizeof(float) + sizeof(uint32_t);
  cloud_msg_.is_dense = false;
  cloud_msg_.is_bigendian = false;
  cloud_msg_.fields[0].name = "x";
  cloud_msg_.fields[0].offset = 0;
  cloud_msg_.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
  cloud_msg_.fields[0].count = 1;
  cloud_msg_.fields[1].name = "y";
  cloud_msg_.fields[1].offset = 4;
  cloud_msg_.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
  cloud_msg_.fields[1].count = 1;
  cloud_msg_.fields[2].name = "z";
  cloud_msg_.fields[2].offset = 8;
  cloud_msg_.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
  cloud_msg_.fields[2].count = 1;
  cloud_msg_.fields[3].name = "rgb";
  cloud_msg_.fields[3].offset = 12;
  cloud_msg_.fields[3].datatype = sensor_msgs::PointField::UINT32;
  cloud_msg_.fields[3].count = 1;
  cloud_msg_.data.resize(33 * cloud_msg_.point_step);
  cloud_msg_.width = 33;
  cloud_msg_.height = 1;
}

// sensor_msgs::PointCloud2& Articulation::PointCloudMsg() {
//   return cloud_msg_;
// }

void Articulation::Print() {
  if (type_ == "prismatic")
    std::cout << "normal_vector: {" << sin(paa_) * cos(aaa_) << ", "
              << sin(paa_) * sin(aaa_) << ", " << cos(paa_) << "}" << std::endl;
  else if (type_ == "revolute") {
    Eigen::Vector3d n = axis_ / axis_.norm();
    std::cout << "normal_vector: {" << n.x() << "," << n.y() << "," << n.z()
              << "}" << std::endl;
  }
  std::cout << "offset vector: {" << offset_.x() << ", " << offset_.y() << ", "
            << offset_.z() << "}" << std::endl;
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

void Articulation::UpdatePointCloud(std::vector<Vector3d> &pc,
                                    const size_t &n) {
  point_cloud_.clear();

  size_t step = pc.size() / (n + 1);
  size_t i = 0;
  avg_.push_back({0,0,0});
  while (point_cloud_.size() < n) {
    point_cloud_.push_back(pc.at(i));
    avg_.back() += pc.at(i);
    i += step;
  }
  avg_.back() /= n;

  // if (first_update) {
  //   first_update = false;
  //   while (point_cloud_.size() < n) {
  //     point_cloud_.push_back(pc.at(i));
  //     avg_ += pc.at(i);
  //     i += step;
  //   }
  //   avg_ /= n;
  // } else {
  //   while (point_cloud_.size() < n) {
  //     point_cloud_.push_back(pc[i]);
  //     i += step;
  //   }
  // }
}

// --------------------- ONLINE OPTIMIZATION FUNCTIONS -------------------------

void Articulation::OptimizePrismaticOnline(const double &time) {
  if (point_cloud_.empty()) {
    return;
  }

  thetas_.push_back(0);
  times_.push_back(time);

  for (size_t j = 0; j < point_cloud_.size(); j++) {
    CostFunction *cf =
        new AutoDiffCostFunction<PrismaticDepthResidual, 1, 1, 1, 1>(
            new PrismaticDepthResidual(point_cloud_[j], avg_[0]));
    problem_.AddResidualBlock(cf, new CauchyLoss(0.5), &paa_, &aaa_,
                              &thetas_.back());
  }

  if (!added_bounds_) {
    problem_.SetParameterLowerBound(&paa_, 0, -1 * M_PI * 2);
    problem_.SetParameterUpperBound(&paa_, 0, M_PI * 2);
    problem_.SetParameterLowerBound(&aaa_, 0, -1 * M_PI * 2);
    problem_.SetParameterUpperBound(&aaa_, 0, M_PI * 2);
    added_bounds_ = true;
  }

  Solve(options_, &problem_, &summary_);

  auto pose1 = Eigen::Isometry3d::Identity();
  visual_tools_->publishArrow(pose1, rvt::RAND);
  visual_tools_->trigger();

  point_cloud_.clear();
}

void Articulation::OptimizeRevoluteOnline(const double &time) {
  if (point_cloud_.empty()) {
    return;
  }

  thetas_.push_back(0);

  for (size_t j = 0; j < point_cloud_.size(); j++) {
    CostFunction *cf = new AutoDiffCostFunction<RevoluteDepthResidual3Norm, 2,
                                                1, 1, 1, 1, 1, 1, 1>(
        new RevoluteDepthResidual3Norm(point_cloud_[j], avg_[0], avg_.back()));
    problem_.AddResidualBlock(cf, new CauchyLoss(0.5), &axis_.x(), &axis_.y(),
                              &axis_.z(), &thetas_.back(), &offset_.x(),
                              &offset_.y(), &offset_.z());
  }

  if (!times_.empty()) {
    CostFunction *cf = new AutoDiffCostFunction<RevoluteMotionResidual, 1, 1, 
                            1>(new RevoluteMotionResidual(times_.back(), time));
    problem_.AddResidualBlock(cf, new CauchyLoss(0.5), 
                              &thetas_[thetas_.size() - 2], &thetas_.back());
  }
    

  times_.push_back(time);

  if (!added_bounds_) {
    problem_.SetParameterLowerBound(&axis_.x(), 0, -2);
    problem_.SetParameterUpperBound(&axis_.x(), 0, 2);
    problem_.SetParameterLowerBound(&axis_.y(), 0, -2);
    problem_.SetParameterUpperBound(&axis_.y(), 0, 2);
    problem_.SetParameterLowerBound(&axis_.z(), 0, -2);
    problem_.SetParameterUpperBound(&axis_.z(), 0, 2);
    added_bounds_ = true;
  }

  problem_.SetParameterLowerBound(&thetas_.back(), 0, -1 * M_PI);
  problem_.SetParameterUpperBound(&thetas_.back(), 0, M_PI);

  Solve(options_, &problem_, &summary_);

  // you need: normal vector of the plane, cacced from theta, p0 and axis
  // the normal arrow should start from p0_t = rodriquez(p0, axis, theta) +
  // offset and end at normalvec + p0_t axis arrow is drawn as already done

  Vector3d phi_axis = axis_ / axis_.norm();
  Vector3d v = avg_[0] - offset_;
  Vector3d p0_t = v * cos(thetas_.back()) +
                  phi_axis.cross(v) * sin(thetas_.back()) +
                  phi_axis * phi_axis.dot(v) * (1. - cos(thetas_.back()));

  Vector3d n = p0_t.cross(phi_axis);
  n /= n.norm();

  p0_t += offset_;

  geometry_msgs::Point norm_start;
  geometry_msgs::Point norm_end;
  norm_start.x = p0_t[0];
  norm_start.y = p0_t[1];
  norm_start.z = p0_t[2];
  norm_end.x = n[0] / 3 + p0_t[0];
  norm_end.y = n[1] / 3 + p0_t[1];
  norm_end.z = n[2] / 3 + p0_t[2];

  geometry_msgs::Point start;
  geometry_msgs::Point end;
  start.x = offset_.x();
  start.y = offset_.y();
  start.z = offset_.z();
  end.x = start.x + axis_.x();
  end.y = start.y + axis_.y();
  end.z = start.z + axis_.z();

  geometry_msgs::Point door1_start;
  geometry_msgs::Point door1_end;
  door1_start.x = p0_t[0];
  door1_start.y = p0_t[1];
  door1_start.z = p0_t[2];
  door1_end.x = offset_.x();
  door1_end.y = offset_.y();
  door1_end.z = offset_.z();

  geometry_msgs::Point door2_start;
  geometry_msgs::Point door2_end;
  door2_start.x = p0_t[0];
  door2_start.y = p0_t[1];
  door2_start.z = p0_t[2];
  door2_end.x = start.x + axis_.x();
  door2_end.y = start.y + axis_.y();
  door2_end.z = start.z + axis_.z();
  // start.x = 0;
  // start.y = 0;
  // start.z = 0;
  // end.x = axis_.x();
  // end.y = axis_.y();
  // end.z = axis_.z();

  // Vector3d plane_normal = n + offset_;

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg_, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg_, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg_, "z");
  sensor_msgs::PointCloud2Iterator<uint32_t> iter_rgb(cloud_msg_, "rgb");

  assert(point_cloud_.size() > 0);
  for (size_t idx = 0; idx < point_cloud_.size(); ++idx) {
    *iter_x = point_cloud_[idx].x();
    *iter_y = point_cloud_[idx].y();
    *iter_z = point_cloud_[idx].z();
    *iter_rgb = 0xC0C0C0LU;
    ++iter_x;
    ++iter_y;
    ++iter_z;
    ++iter_rgb;
  }

  cloud_publisher_->publish(cloud_msg_);

  visual_tools_->deleteAllMarkers();
  visual_tools_->publishLine(door1_start, door1_end, rvt::BLUE);
  visual_tools_->publishLine(door2_start, door2_end, rvt::BLUE);
  visual_tools_->publishArrow(norm_start, norm_end, rvt::RED);
  visual_tools_->publishArrow(start, end, rvt::BLUE);
  visual_tools_->trigger();

  point_cloud_.clear();
}
} // namespace articulation
