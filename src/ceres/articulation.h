#include <vector>

#include "eigen3/Eigen/Dense"
#include "ceres/ceres.h"

#ifndef ARTICULATION_H
#define ARTICULATION_H

namespace ros {
  class NodeHandle;
}  // namespace ros

namespace articulation {

class Articulation {
 public:

   // Constructor
  explicit Articulation(ros::NodeHandle* n, size_t num_points);

  void Run();
  void UpdatePointCloud(std::vector<Eigen::Vector3f>& pc);
  void Optimize();
  void OptimizeDepthPrismatic();
  
  bool optimizer_ready_;

 private:

  std::vector<std::vector<Eigen::Vector3f>> point_clouds_;
  Eigen::Vector3f avg_;
  size_t num_points_;
};

struct RevoluteDepthResidual {
  const double x_;
  const double y_;
  const double z_;
  const double p0_x_;
  const double p0_y_;
  const double p0_z_;


  RevoluteDepthResidual(double p0x, double p0y, double p0z, double x, double y, double z) :
      x_(x),
      y_(y),
      z_(z),
      p0_x_(p0x),
      p0_y_(p0y),
      p0_z_(p0z) {}

  template <typename T>
  bool operator()(const T* const paa, const T* const aaa, const T* const theta, const T* const o_x, const T* const o_y, const T* const o_z, T* residual) const {
    // const Eigen::Matrix<T, 3, 1> offset = {*o_x, *o_y, *o_z};
    // const auto v_x = p0_x_ - *o_x;
    // const auto v_y = p0_y_ - *o_y;
    // const auto v_z = p0_z_ - *o_z;

    // const auto k_x = sin(*paa) * cos(*aaa);
    // const auto k_y = sin(*paa) * sin(*aaa);
    // const auto k_z = cos(*paa);

    // const auto v_rot_x = v_x*cos(*theta) + (k_y*v_z-k_z*v_y)*sin(*theta)+k_x*(k_x*v_x+k_y*v_y+k_z*v_z)*(1.-cos(*theta));
    // const auto v_rot_y = v_y*cos(*theta) + (k_z*v_x-k_x*v_z)*sin(*theta)+k_y*(k_x*v_x+k_y*v_y+k_z*v_z)*(1.-cos(*theta));
    // const auto v_rot_z = v_z*cos(*theta) + (k_x*v_y-k_y*v_x)*sin(*theta)+k_z*(k_x*v_x+k_y*v_y+k_z*v_z)*(1.-cos(*theta));

    // const auto normal_x = k_y*v_rot_z-k_z*v_rot_y;
    // const auto normal_y = k_z*v_rot_x-k_x*v_rot_z;
    // const auto normal_z = k_x*v_rot_y-k_y*v_rot_x;

    // const auto unorm_x = normal_x / sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z);
    // const auto unorm_y = normal_y / sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z);
    // const auto unorm_z = normal_z / sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z);

    // residual[0] = unorm_x*(x_ - *o_x)+unorm_y*(y_ - *o_y)+unorm_z*(z_ - *o_z);

    const Eigen::Matrix<T, 3, 1> v = {
      p0_x_ - *o_x,
      p0_y_ - *o_y, 
      p0_z_ - *o_z
    };
    const Eigen::Matrix<T, 3, 1> phi_axis = {
      sin(*paa) * cos(*aaa),
      sin(*paa) * sin(*aaa),
      cos(*paa)
    };
    const Eigen::Matrix<T, 3, 1> v_rot = v*cos(*theta) + 
                                         phi_axis.cross(v)*sin(*theta) + 
                                         phi_axis*phi_axis.dot(v)*(1.-cos(*theta));
    const Eigen::Matrix<T, 3, 1> crossnormal = phi_axis.cross(v_rot);
    const Eigen::Matrix<T, 3, 1> normal = crossnormal / crossnormal.norm();
    const Eigen::Matrix<T, 3, 1> p = {x_ - *o_x, y_ - *o_y, z_ - *o_z};

    residual[0] = normal.dot(p);
    return true;
  } 
};  

struct PrismaticDepthResidual {
  const double x_;
  const double y_;
  const double z_;
  const double p0_x_;
  const double p0_y_;
  const double p0_z_;


  PrismaticDepthResidual(double p0x, double p0y, double p0z, double x, double y, double z) :
      x_(x),
      y_(y),
      z_(z),
      p0_x_(p0x),
      p0_y_(p0y),
      p0_z_(p0z) {}

  template <typename T>
  bool operator()(const T* const paa, const T* const aaa, const T* const theta, T* residual) const {
    // p_plane is an adjusted p0, we have to multiply normal with theta diff
    // normal(paa, aaa)
    // norm_x = sin(*paa) * cos(*aaa);
    // norm_y = sin(*paa) * sin(*aaa);
    // norm_z = cos(*paa);
    // (p0 + theta*normal(paa, aaa)) = p_plane
    // residual = normal(paa, aaa) dot (p_plane - {x,y,z})
    // dot_x = (sin(*paa) * cos(*aaa) * (*theta) + p0_x_ - x_) * sin(*paa) * cos(*aaa);
    // dot_y = (sin(*paa) * sin(*aaa) * (*theta) + p0_y_ - y_) * sin(*paa) * sin(*aaa);
    // dot_z = (cos(*paa) * (*theta) + p0_z_ - z_) * cos(*paa);
    // residual[0] = abs(dot_x + dot_y + dot_z)
    residual[0] = abs((sin(*paa) * cos(*aaa) * (*theta) + p0_x_ - x_) * sin(*paa) * cos(*aaa) + (sin(*paa) * sin(*aaa) * (*theta) + p0_y_ - y_) * sin(*paa) * sin(*aaa) + (cos(*paa) * (*theta) + p0_z_ - z_) * cos(*paa));
    return true;
  } 
};

}  // namespace navigation

#endif  // NAVIGATION_H
