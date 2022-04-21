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
  explicit Articulation(ros::NodeHandle *n, const size_t& num_iters, const std::string& type);

  void UpdatePointCloud(std::vector<Eigen::Vector3d> &pc, const size_t& n);

  void OptimizeRevoluteOnline(const double& time);
  void OptimizePrismaticOnline(const double& time);

  void Print();

private:
  std::vector<Eigen::Vector3d> point_cloud_;
  Eigen::Vector3d avg_;

  bool added_bounds_ = false;
  double paa_ = 0;
  double aaa_ = 0;
  double n_x_ = 0;
  double n_y_ = 0;
  double n_z_ = 1;
  double offset_x_ = 0;
  double offset_y_ = 0;
  double offset_z_ = 0;
  std::vector<double> thetas_;
  std::vector<double> times_;
  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
  bool first_update = true;
  std::string type_;
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

struct RevoluteDepthResidual3Norm {
  const Eigen::Vector3d p_;
  const Eigen::Vector3d p0_;

  RevoluteDepthResidual3Norm(const Eigen::Vector3d &p0, const Eigen::Vector3d &p)
      : p_(p), p0_(p0) {}

  template <typename T>
  bool operator()(const T *const n_x, const T *const n_y, const T *const n_z, const T *const theta,
                  const T *const o_x, const T *const o_y, const T *const o_z,
                  T *residual) const {

    const Eigen::Matrix<T, 3, 1> phi = {*n_x, *n_y, *n_z};
    const Eigen::Matrix<T, 3, 1> phi_axis = phi / phi.norm();

    const Eigen::Matrix<T, 3, 1> offset = {*o_x, *o_y, *o_z};
    const Eigen::Matrix<T, 3, 1> v = p0_ - offset;

    const Eigen::Matrix<T, 3, 1> v_rot =
        v * cos(*theta) + phi_axis.cross(v) * sin(*theta) +
        phi_axis * phi_axis.dot(v) * (1. - cos(*theta));

    // const Eigen::Matrix<T, 3, 1> proj_par = v.dot(phi_axis) / phi_axis.dot(phi_axis) * phi_axis;
    // const Eigen::Matrix<T, 3, 1> proj_perp = v - proj_par;
    // const Eigen::Matrix<T, 3, 1> wnn = phi_axis.cross(proj_perp);
    // const Eigen::Matrix<T, 3, 1> w = wnn / wnn.norm();
    // const Eigen::Matrix<T, 3, 1> v_rot = proj_par + proj_perp * cos(*theta) + proj_perp.norm() * w * sin(*theta);

    const Eigen::Matrix<T, 3, 1> crossnormal = phi_axis.cross(v_rot);
    const Eigen::Matrix<T, 3, 1> normal = crossnormal / crossnormal.norm();
    const Eigen::Matrix<T, 3, 1> p = p_ - offset;

    residual[0] = abs(normal.dot(p)) + abs(phi.norm() - 1.0);
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
