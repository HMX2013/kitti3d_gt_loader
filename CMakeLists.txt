cmake_minimum_required(VERSION 3.0.2)
project(kitti3d)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        roslaunch
        cv_bridge
        pcl_conversions
        pcl_ros
        geometry_msgs
        laser_geometry
        sensor_msgs
        obsdet_msgs
        )

find_package(PCL 1.7 REQUIRED)
find_package(Boost 1.54 REQUIRED)

catkin_package(
        INCLUDE_DIRS
        LIBRARIES
        CATKIN_DEPENDS roscpp rospy std_msgs obsdet_msgs
)

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        include
)

add_executable(ros_kitti3d_array src/ros_kitti3d_array.cpp)
target_link_libraries(ros_kitti3d_array ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(ros_kitti3d_array
        ${catkin_EXPORTED_TARGETS}
)