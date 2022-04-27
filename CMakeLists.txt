PROJECT(reference_car)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

MESSAGE(STATUS "Compiling Reference Car Localization + Navigation")
MESSAGE(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")

SET(CMAKE_CXX_FLAGS "-Wall -Werror")

IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
  MESSAGE(STATUS "Additional Flags for Release mode")
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -O2 -DNDEBUG")
ELSEIF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  MESSAGE(STATUS "Additional Flags for Debug mode")
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")
ENDIF()

INCLUDE($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
ROSBUILD_INIT()
SET(ROS_BUILD_STATIC_LIBS true)
SET(ROS_BUILD_SHARED_LIBS false)
FIND_PACKAGE(Ceres REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)

MESSAGE(STATUS "ROS-Overrride Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

SET(libs roslib roscpp rosbag glog gflags boost_system lua5.1
    pthread)
#             src/visualization/visualization.cc
#             src/vector_map/vector_map.cc)

INCLUDE_DIRECTORIES(src/shared)
ADD_SUBDIRECTORY(src/shared)
INCLUDE_DIRECTORIES(src)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

# ROSBUILD_ADD_EXECUTABLE(particle_filter
#                         src/particle_filter/particle_filter_main.cc
#                         src/particle_filter/particle_filter.cc)
# TARGET_LINK_LIBRARIES(particle_filter shared_library ${libs})

# ROSBUILD_ADD_EXECUTABLE(navigation
#                         src/navigation/navigation_main.cc
#                         src/navigation/navigation.cc)
# TARGET_LINK_LIBRARIES(navigation shared_library ${libs})


ROSBUILD_ADD_EXECUTABLE(articulation
                        src/ceres/articulation_main.cc
                        src/ceres/articulation.cc)
TARGET_LINK_LIBRARIES(articulation amrl_shared_lib rviz_visual_tools ${libs} ${CERES_INCLUDE_DIRS} ceres)


# ADD_EXECUTABLE(eigen_tutorial
#                src/eigen_tutorial.cc)

# ADD_EXECUTABLE(simple_queue_test
#                src/navigation/simple_queue_test.cc)