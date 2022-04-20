#include <vector>

#include "ceres/ceres.h"
#include "eigen3/Eigen/Dense"

#ifndef ARTICULATION_H
#define ARTICULATION_H

namespace ros {
class NodeHandle;
} // namespace ros

namespace articulation {

class Articulation {
public:
  // Constructor
  explicit Articulation(ros::NodeHandle *n);

  void UpdatePointCloud(std::vector<Eigen::Vector3d> &pc);
  
  void OptimizeRevoluteOffline();
  void OptimizePrismaticOffline();
  void OptimizeRevoluteOnline();
  void OptimizePrismaticOnline();

  void Print();

private:
  std::vector<Eigen::Vector3d> point_cloud_;
  Eigen::Vector3d avg_;

  bool added_bounds_ = false;
  double paa_ = 0;
  double aaa_ = 0;
  double offset_x_ = 0;
  double offset_y_ = 0;
  double offset_z_ = 0;
  std::vector<double> thetas_;
  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
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
