#include <vector>

#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"
#include "sensor_msgs/PointCloud2.h"
#include <rviz_visual_tools/rviz_visual_tools.h>

#ifndef ARTICULATION_H
#define ARTICULATION_H

namespace ros {
class NodeHandle;
} // namespace ros

namespace articulation {

class Articulation {
public:
  // Constructor
  explicit Articulation(ros::NodeHandle *n, const size_t &num_iters,
                        const std::string &type,
                        ros::Publisher *cloud_publisher);

  void UpdatePointCloud(std::vector<Eigen::Vector3d> &pc, const size_t &n);

  void OptimizeRevoluteOnline(const double &time);
  void OptimizePrismaticOnline(const double &time);

  // sensor_msgs::PointCloud2& PointCloudMsg();

  void Print();

private:
  std::vector<Eigen::Vector3d> point_cloud_;
  std::vector<Eigen::Vector3d> avg_;

  bool added_bounds_ = false;
  double paa_ = 0;
  double aaa_ = 0;
  sensor_msgs::PointCloud2 cloud_msg_;
  Eigen::Vector3d axis_;
  Eigen::Vector3d offset_;
  std::vector<double> thetas_;
  std::vector<double> times_;
  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
  bool first_update = true;
  std::string type_;
  rviz_visual_tools::RvizVisualToolsPtr visual_tools_;
  ros::Publisher *cloud_publisher_;
};

struct RevoluteDepthResidual {
  const Eigen::Vector3d p_;
  const Eigen::Vector3d p0_;
  

  RevoluteDepthResidual(const Eigen::Vector3d &p0, const Eigen::Vector3d &p)
      : p_(p), p0_(p0) {}

  template <typename T>
  bool operator()(const T *const paa, const T *const aaa, const T *const theta,
                  const T *const o_x, const T *const o_y, const T *const o_z,
                  T *residual) const {
    const Eigen::Matrix<T, 3, 1> offset = {*o_x, *o_y, *o_z};
    const Eigen::Matrix<T, 3, 1> v = p0_ - offset;
    const Eigen::Matrix<T, 3, 1> phi_axis = {sin(*paa) * cos(*aaa),
                                             sin(*paa) * sin(*aaa), cos(*paa)};
    const Eigen::Matrix<T, 3, 1> v_rot =
        v * cos(*theta) + phi_axis.cross(v) * sin(*theta) +
        phi_axis * phi_axis.dot(v) * (1. - cos(*theta));
    const Eigen::Matrix<T, 3, 1> crossnormal = phi_axis.cross(v_rot);
    const Eigen::Matrix<T, 3, 1> normal = crossnormal / crossnormal.norm();
    const Eigen::Matrix<T, 3, 1> p = p_ - offset;

    residual[0] = abs(normal.dot(p));
    return true;
  }
};

struct RevoluteMotionResidual {
  const double prev_time_;
  const double curr_time_;

  RevoluteMotionResidual(const double& prev_time,
                         const double& curr_time) : prev_time_(prev_time), curr_time_(curr_time) {} 

  template <typename T>
  bool operator()(const T *const prev_theta, const T *const curr_theta, T *residual) const {
    const T ang_vel = (*curr_theta - *prev_theta) / (curr_time_ - prev_time_);
    
    // if the instantaneous rate is greater than 0.6 rad/sec or 34 deg/sec
    if (ang_vel > 0.6) {
      residual[0] = abs(*curr_theta - *prev_theta);
    } else {
      residual[0] = abs(*curr_theta - *curr_theta);
    }
    return true;
  }
};

struct RevoluteDepthResidual3Norm {
  const Eigen::Vector3d p_;
  const Eigen::Vector3d p0_;
  const Eigen::Vector3d avg_t_;

  RevoluteDepthResidual3Norm(const Eigen::Vector3d &p0,
                             const Eigen::Vector3d &p,
                             const Eigen::Vector3d &avg)
      : p_(p), p0_(p0), avg_t_(avg) {}

  template <typename T>
  bool operator()(const T *const n_x, const T *const n_y, const T *const n_z,
                  const T *const theta, const T *const o_x, const T *const o_y,
                  const T *const o_z, T *residual) const {

    const Eigen::Matrix<T, 3, 1> phi = {*n_x, *n_y, *n_z};
    const Eigen::Matrix<T, 3, 1> phi_axis = phi / phi.norm();

    const Eigen::Matrix<T, 3, 1> offset = {*o_x, *o_y, *o_z};
    const Eigen::Matrix<T, 3, 1> v = p0_ - offset;

    const T theta1 = -(*theta);

    const Eigen::Matrix<T, 3, 1> v_rot =
        v * cos(theta1) + phi_axis.cross(v) * sin(theta1) +
        phi_axis * phi_axis.dot(v) * (1. - cos(theta1));

    const Eigen::Matrix<T, 3, 1> crossnormal = v_rot.cross(phi_axis);
    const Eigen::Matrix<T, 3, 1> normal = crossnormal / crossnormal.norm();
    const Eigen::Matrix<T, 3, 1> p = p_ - offset;

    residual[0] = abs(normal.dot(p));
    residual[1] = abs(phi.norm() - 1.0); 
    return true;
  }
};

struct PrismaticDepthResidual {
  const Eigen::Vector3d p_;
  const Eigen::Vector3d p0_;

  PrismaticDepthResidual(const Eigen::Vector3d &p0, const Eigen::Vector3d &p)
      : p_(p), p0_(p0) {}

  template <typename T>
  bool operator()(const T *const n_x, const T *const n_y, const T *const theta,
                  T *residual) const {
    const Eigen::Matrix<T, 3, 1> normal = {sin(*n_x) * cos(*n_y),
                                           sin(*n_x) * sin(*n_y), cos(*n_x)};

    residual[0] = abs(normal.dot(p0_ + normal * (*theta) - p_));
    return true;
  }
};

} // namespace articulation

#endif // NAVIGATION_H
