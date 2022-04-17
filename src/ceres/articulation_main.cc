#include <signal.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <vector>
#include <fstream>

#include "glog/logging.h"
#include "gflags/gflags.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "amrl_msgs/Localization2DMsg.h"
#include "gflags/gflags.h"
#include "geometry_msgs/Pose2D.h"
#include "geometry_msgs/PoseArray.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "sensor_msgs/LaserScan.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "nav_msgs/Odometry.h"
#include "ros/ros.h"
#include "shared/math/math_util.h"
#include "shared/util/timer.h"
#include "shared/ros/ros_helpers.h"
#include "config_reader/config_reader.h"

#include "std_msgs/String.h"

#include "articulation.h"

using amrl_msgs::Localization2DMsg;
using ros::Time;
using ros_helpers::Eigen3DToRosPoint;
using articulation::Articulation;
using ros_helpers::Eigen2DToRosPoint;
using ros_helpers::RosPoint;
using ros_helpers::SetRosVector;
using std::string;
using std::vector;
using Eigen::Vector2f;
using Eigen::Vector3f;

CONFIG_FLOAT(x_min_, "x_min");
CONFIG_FLOAT(x_max_, "x_max");
CONFIG_FLOAT(y_min_, "y_min");
CONFIG_FLOAT(y_max_, "y_max");
CONFIG_FLOAT(z_min_, "z_min");
CONFIG_FLOAT(z_max_, "z_max");
CONFIG_UINT(num_points_, "num_points");
config_reader::ConfigReader config_reader_({"config/articulation.lua"});

int ddx = 0;
DEFINE_string(points_topic, "kinect_points", "Name of ROS Topic");
bool run_ = true;
Articulation* articulation_ = nullptr;

float ToFloat(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
  float f;
  uint8_t b[] = {b3, b2, b1, b0};
  memcpy(&f, &b, sizeof(f));
  return f;
}

void KinectCallback(const sensor_msgs::PointCloud2& cloud_msg) {
  // ddx += 1;
  // if (ddx != 3)
  //   return;
  // std::cout <<  cloud_msg.fields[0].name << " " << cloud_msg.fields[0].offset << " |" << static_cast<uint8_t>(cloud_msg.fields[0].datatype) << "| " << cloud_msg.fields[0].count << std::endl;
  // unsigned char* p = (unsigned char*)&cloud_msg.is_bigendian;
  // printf("datatype: %x\n", p[0]);
  
  std::vector<Vector3f> point_cloud;
  for (uint32_t i = 0; i < cloud_msg.height; i++) {
    for (uint32_t j = 0; j < cloud_msg.width*cloud_msg.point_step; j+=cloud_msg.point_step) {
      size_t idx = i*cloud_msg.width*cloud_msg.point_step+j;
      float x = ToFloat(cloud_msg.data[idx+3], cloud_msg.data[idx+2], cloud_msg.data[idx+1], cloud_msg.data[idx+0]);
      if (!(x > CONFIG_x_min_ and x < CONFIG_x_max_) or x == 0)
        continue;
      float y = ToFloat(cloud_msg.data[idx+7], cloud_msg.data[idx+6], cloud_msg.data[idx+5], cloud_msg.data[idx+4]);
      if (!(y > CONFIG_y_min_ and y < CONFIG_y_max_) or y == 0)
        continue;
      float z = ToFloat(cloud_msg.data[idx+11], cloud_msg.data[idx+10], cloud_msg.data[idx+9], cloud_msg.data[idx+8]);
      if (!(z > CONFIG_z_min_ and z < CONFIG_z_max_))
        continue;
      point_cloud.push_back({x,y,z});
    }
  }

  articulation_->UpdatePointCloud(point_cloud);

  // std::string filename = "./rev3.csv";
  // std::ofstream fout(filename);
  // int ac = 0;
  // for (auto it : point_cloud) {
  //   if (!(it(0) == 0 && it(1) == 0)) {
  //     ac+=1;
  //     fout << it(0) << "," << it(1) << "," << it(2) << "," << std::endl;
  //   }
  // }
  // fout.close();
}

void SignalHandler(int) {
  if (!run_) {
    printf("Force Exit.\n");
    exit(0);
  }
  printf("Exiting.\n");
  run_ = false;
}

int main(int argc, char** argv) {
  ddx = 0;
  google::ParseCommandLineFlags(&argc, &argv, false);
  signal(SIGINT, SignalHandler);
  // Initialize ROS.
  ros::init(argc, argv, "depth_factor", ros::init_options::NoSigintHandler);
  ros::NodeHandle n;
  articulation_ = new Articulation(&n, CONFIG_num_points_);

  ros::Subscriber kinect_sub =
      n.subscribe(FLAGS_points_topic, 1, &KinectCallback);

  // ros::Subscriber scan_sub = 
  //     n.subscribe("kinect_scan", 1, &ScanCallback);

  RateLoop loop(20.0);
  while (run_ && ros::ok() && !articulation_->optimizer_ready_) {
    ros::spinOnce();
    loop.Sleep();
  }

  articulation_->OptimizeDepthPrismatic();

  delete articulation_;
  return 0;
}
