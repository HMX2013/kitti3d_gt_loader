// For disable PCL complile lib, to use PointXYZILID
#define PCL_NO_PRECOMPILE
#include <visualization_msgs/Marker.h>
#include "tools/kitti_loader.hpp"
#include <signal.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <jsk_recognition_msgs/BoundingBox.h>
#include "obsdet_msgs/CloudCluster.h"
#include "obsdet_msgs/CloudClusterArray.h"
#include <opencv2/opencv.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

using PointType = pcl::PointXYZ;

using namespace std;

ros::Publisher CloudPublisher;
ros::Publisher ClusterPublisher;
ros::Publisher ClusterArrayPublisher;
ros::Publisher pub_jsk_bboxs_gt_;
ros::Publisher pub_angle_valid_pt2_;
ros::Publisher pub_closest_rec_pt_;

std::string output_filename;
std::string acc_filename;
std::string pcd_savepath;
std::string data_path;
std::string label_path;
std::string debug_path;
std::string seq;
std::string frame_id_;

int debug_seqn;
int debug_seqn_2;

double rate_hz;
bool save_flag;
bool debug;

void signal_callback_handler(int signum) {
    cout << "Caught Ctrl + c " << endl;
    // Terminate program
    exit(signum);
}

template<typename T>
sensor_msgs::PointCloud2 cloud2msg(pcl::PointCloud<T> cloud, std::string frame_id = "map") {
    sensor_msgs::PointCloud2 cloud_ROS;
    pcl::toROSMsg(cloud, cloud_ROS);
    cloud_ROS.header.frame_id = frame_id;
    return cloud_ROS;
}

void calculateDimPos(double &theta_star, jsk_recognition_msgs::BoundingBox &output, pcl::PointCloud<pcl::PointXYZ> &cluster)
{
  // calc centroid point for cylinder height(z)
  pcl::PointXYZ centroid;
  centroid.x = 0;
  centroid.y = 0;
  centroid.z = 0;

  for (const auto &pcl_point : cluster)
  {
    centroid.x += pcl_point.x;
    centroid.y += pcl_point.y;
    centroid.z += pcl_point.z;
  }

  centroid.x = centroid.x / (double)cluster.size();
  centroid.y = centroid.y / (double)cluster.size();
  centroid.z = centroid.z / (double)cluster.size();

  // calc min and max z for cylinder length
  double min_z = 0;
  double max_z = 0;

  for (size_t i = 0; i < cluster.size(); ++i)
  {
    if (cluster.at(i).z < min_z || i == 0)
      min_z = cluster.at(i).z;
    if (max_z < cluster.at(i).z || i == 0)
      max_z = cluster.at(i).z;
  }

  Eigen::Vector2d e_1_star;  // col.11, Algo.2
  Eigen::Vector2d e_2_star;
  e_1_star << std::cos(theta_star), std::sin(theta_star);
  e_2_star << -std::sin(theta_star), std::cos(theta_star);
  std::vector<double> C_1_star;  // col.11, Algo.2
  std::vector<double> C_2_star;  // col.11, Algo.2
  for (const auto& point : cluster)
  {
    C_1_star.push_back(point.x * e_1_star.x() + point.y * e_1_star.y());
    C_2_star.push_back(point.x * e_2_star.x() + point.y * e_2_star.y());
  }

  // col.12, Algo.2
  const double min_C_1_star = *std::min_element(C_1_star.begin(), C_1_star.end());
  const double max_C_1_star = *std::max_element(C_1_star.begin(), C_1_star.end());
  const double min_C_2_star = *std::min_element(C_2_star.begin(), C_2_star.end());
  const double max_C_2_star = *std::max_element(C_2_star.begin(), C_2_star.end());

  const double a_1 = std::cos(theta_star);
  const double b_1 = std::sin(theta_star);
  const double c_1 = min_C_1_star;

  const double a_2 = -1.0 * std::sin(theta_star);
  const double b_2 = std::cos(theta_star);
  const double c_2 = min_C_2_star;

  const double a_3 = std::cos(theta_star);
  const double b_3 = std::sin(theta_star);
  const double c_3 = max_C_1_star;

  const double a_4 = -1.0 * std::sin(theta_star);
  const double b_4 = std::cos(theta_star);
  const double c_4 = max_C_2_star;

  // calc center of bounding box
  double intersection_x_1 = (b_1 * c_2 - b_2 * c_1) / (a_2 * b_1 - a_1 * b_2);
  double intersection_y_1 = (a_1 * c_2 - a_2 * c_1) / (a_1 * b_2 - a_2 * b_1);
  double intersection_x_2 = (b_3 * c_4 - b_4 * c_3) / (a_4 * b_3 - a_3 * b_4);
  double intersection_y_2 = (a_3 * c_4 - a_4 * c_3) / (a_3 * b_4 - a_4 * b_3);

  // calc dimention of bounding box
  Eigen::Vector2d e_x;
  Eigen::Vector2d e_y;
  e_x << a_1 / (std::sqrt(a_1 * a_1 + b_1 * b_1)), b_1 / (std::sqrt(a_1 * a_1 + b_1 * b_1));
  e_y << a_2 / (std::sqrt(a_2 * a_2 + b_2 * b_2)), b_2 / (std::sqrt(a_2 * a_2 + b_2 * b_2));
  Eigen::Vector2d diagonal_vec;
  diagonal_vec << intersection_x_1 - intersection_x_2, intersection_y_1 - intersection_y_2;

  // calc yaw
  tf2::Quaternion quat;
  quat.setEuler(/* roll */ 0, /* pitch */ 0, /* yaw */ std::atan2(e_1_star.y(), e_1_star.x()));

  output.pose.position.x = (intersection_x_1 + intersection_x_2) / 2.0;
  output.pose.position.y = (intersection_y_1 + intersection_y_2) / 2.0;
  output.pose.position.z = centroid.z;

  output.pose.orientation = tf2::toMsg(quat);
  constexpr double ep = 0.001;
  output.dimensions.x = std::fabs(e_x.dot(diagonal_vec));
  output.dimensions.y = std::fabs(e_y.dot(diagonal_vec));
  output.dimensions.z = std::max((max_z - min_z), ep);

  output.dimensions.x = std::max(output.dimensions.x, ep);
  output.dimensions.y = std::max(output.dimensions.y, ep);

  // calculate the closest pt
  cv::Point2f rec_corner_p1, rec_corner_p2, rec_corner_p3, rec_corner_p4;
  std::vector<cv::Point2f> rec_corner_points;

  rec_corner_p1.x = (b_1 * c_2 - b_2 * c_1) / (a_2 * b_1 - a_1 * b_2);
  rec_corner_p1.y = (a_1 * c_2 - a_2 * c_1) / (a_1 * b_2 - a_2 * b_1);

  rec_corner_p2.x = (b_2 * c_3 - b_3 * c_2) / (a_3 * b_2 - a_2 * b_3);
  rec_corner_p2.y = (a_2 * c_3 - a_3 * c_2) / (a_2 * b_3 - a_3 * b_2);

  rec_corner_p3.x = (b_3 * c_4 - b_4 * c_3) / (a_4 * b_3 - a_3 * b_4);
  rec_corner_p3.y = (a_3 * c_4 - a_4 * c_3) / (a_3 * b_4 - a_4 * b_3);

  rec_corner_p4.x = (b_1 * c_4 - b_4 * c_1) / (a_4 * b_1 - a_1 * b_4);
  rec_corner_p4.y = (a_1 * c_4 - a_4 * c_1) / (a_1 * b_4 - a_4 * b_1);

  rec_corner_points.push_back(rec_corner_p1);
  rec_corner_points.push_back(rec_corner_p2);
  rec_corner_points.push_back(rec_corner_p3);
  rec_corner_points.push_back(rec_corner_p4);

  double min_dist_reccp = std::numeric_limits<double>::max();
  int min_reccp_index;

  for (int i = 0; i < rec_corner_points.size(); ++i)
  {
    if ((rec_corner_points[i].x * rec_corner_points[i].x + rec_corner_points[i].y * rec_corner_points[i].y) < min_dist_reccp || i == 0)
    {
      min_dist_reccp = rec_corner_points[i].x * rec_corner_points[i].x + rec_corner_points[i].y * rec_corner_points[i].y;
      min_reccp_index = i;
    }
  }
}

void kitti2us(double &theta_kitti, double &theta_trans)
{
  if (theta_kitti < 0)
  {
    theta_kitti = M_PI + theta_kitti;
  }

  if (theta_kitti > M_PI / 2)
    theta_trans = M_PI - theta_kitti;
  else
    theta_trans = M_PI / 2 - theta_kitti;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "Ros-Kitti-Publisher");
  ros::NodeHandle nh;
  nh.param<string>("/seq", seq, "00");
  nh.param<double>("/rate_hz", rate_hz, 10);
  nh.param<bool>("/debug", debug, false);
  nh.param<string>("/data_path", data_path, "/");
  nh.param<string>("/label_path", label_path, "/");
  nh.param<string>("/debug_path", debug_path, "/");
  nh.param<string>("/frame_id", frame_id_, "map");

  nh.param<int>("/debug_seqn", debug_seqn, 10);
  nh.param<int>("/debug_seqn_2", debug_seqn_2, 10);

  tf::TransformBroadcaster tfBroadcaster2;
  tf::StampedTransform MapBoxTrans;

  MapBoxTrans.frame_id_ = "map";
  MapBoxTrans.child_frame_id_ = "box_center";

  ros::Rate r(rate_hz);

  CloudPublisher = nh.advertise<sensor_msgs::PointCloud2>("/kitti3d/cloud", 100, true);
  ClusterPublisher = nh.advertise<obsdet_msgs::CloudCluster>("/kitti3d/cluster", 100, true);
  ClusterArrayPublisher = nh.advertise<obsdet_msgs::CloudClusterArray>("/kitti3d/cluster_array", 100, true);
  pub_jsk_bboxs_gt_ = nh.advertise<jsk_recognition_msgs::BoundingBox>("/kitti3d/jsk_bbox_gt", 100);
  pub_angle_valid_pt2_ = nh.advertise<sensor_msgs::PointCloud2>("/kitti3d/lim_angle_pt", 100, true);
  pub_closest_rec_pt_ = nh.advertise<sensor_msgs::PointCloud2>("/kitti3d/closest_rec_pt", 100, true);

  signal(SIGINT, signal_callback_handler);


  KittiLoader loader(data_path, label_path);

  std::vector<int> debug_seq;

  std::vector<int> debug_seq_2;

  loader.debug(debug_path, debug_seq, debug_seq_2);

  int N;

  if (debug){
    N = debug_seq.size();
  }
  else{
    N = loader.size();
  }

  std::vector<std::string> labeled_bin;
  std::vector<double> theta_kitti;
  std::cerr << "\033[1;32m[Kitti Publisher] Total " << N << " clouds are loaded\033[0m" << std::endl;

  obsdet_msgs::CloudClusterArray cluster_gt_array;

  int iter_num;
  if (debug){
    iter_num = 10000;
  }
  else{
    iter_num = N;
  }

  for (int n = 0; n < N; n++)
  {
    const auto start_time = std::chrono::steady_clock::now();
    // std::cout << n << "th node is published!" << endl;

    labeled_bin.clear();
    theta_kitti.clear();

    if (debug){
      loader.get_gt_label(debug_seq[n], debug_seq_2[n], labeled_bin, theta_kitti);
      // loader.get_gt_label(debug_seqn, debug_seqn_2, labeled_bin, theta_kitti);
    }
    else{
      loader.get_gt_label(n, -1, labeled_bin, theta_kitti);
    }

    for (size_t i = 0; i < labeled_bin.size(); i++)
    {
      std::cout <<"labeled_bin is "<< labeled_bin[i] << std::endl;
      // std::cout <<"theta_kitti is "<< theta_kitti[i] << std::endl;
      obsdet_msgs::CloudCluster cluster_gt;

      pcl::PointCloud<PointType> pc_curr;
      loader.get_cloud(labeled_bin[i], pc_curr);

      Eigen::Vector4f centroid_3d;
      pcl::compute3DCentroid(pc_curr, centroid_3d);

      if (pc_curr.size() < 20){
        continue;
      }

      ROS_INFO("centroid_3d[0]=%f",centroid_3d[0]);
      MapBoxTrans.stamp_ = ros::Time::now();
      MapBoxTrans.setRotation(tf::Quaternion(0, 0, 0, 1));
      MapBoxTrans.setOrigin(tf::Vector3(centroid_3d[0], centroid_3d[1], 0));
      tfBroadcaster2.sendTransform(MapBoxTrans);


      double theta_trans;
      jsk_recognition_msgs::BoundingBox jsk_object_gt;

      kitti2us(theta_kitti[i], theta_trans);

      calculateDimPos(theta_trans, jsk_object_gt, pc_curr);

      jsk_object_gt.header.frame_id = frame_id_;
      pub_jsk_bboxs_gt_.publish(jsk_object_gt);

      sensor_msgs::PointCloud2 cloud_msg;
      cloud_msg.header.frame_id = frame_id_;
      pcl::toROSMsg(pc_curr, cloud_msg);

      cluster_gt.header.frame_id = frame_id_;
      cluster_gt.cloud = cloud_msg;
      cluster_gt.orientation = theta_trans;

      if (debug){
        cluster_gt.index = debug_seq[n];
        cluster_gt.index_seq = debug_seq_2[n];
      }
      else{
        cluster_gt.index = n;
        cluster_gt.index_seq = i;
      }

      CloudPublisher.publish(cloud2msg(pc_curr));
      ClusterPublisher.publish(cluster_gt);
      cluster_gt_array.clusters.push_back(cluster_gt);

      cluster_gt_array.header.frame_id = frame_id_;
      cluster_gt_array.header.stamp = ros::Time::now();
      ClusterArrayPublisher.publish(cluster_gt_array);
      cluster_gt_array.clusters.clear();

      // project the 3D cluster pc into 2D pc
      std::vector<cv::Point2f> pc_curr_2d;

      for (unsigned int i = 0; i < pc_curr.points.size(); i++)
      {
        cv::Point2f pt;
        pt.x = pc_curr.points[i].x;
        pt.y = pc_curr.points[i].y;
        pc_curr_2d.push_back(pt);
      }

      std::vector<cv::Point2f> convex_hull_pt;
      cv::convexHull(pc_curr_2d, convex_hull_pt);
      double angle_valid;
      double min_angle = std::numeric_limits<double>::max();
      double max_angle = std::numeric_limits<double>::min();

      int min_index;
      int max_index;

      for (int i = 0; i < convex_hull_pt.size(); i++)
      {
        angle_valid = atan2(convex_hull_pt[i].y, convex_hull_pt[i].x);
        if (angle_valid < min_angle)
        {
          min_angle = angle_valid;
          min_index = i;
        }
        if (angle_valid > max_angle)
        {
          max_angle = angle_valid;
          max_index = i;
        }
      }

      pcl::PointCloud<pcl::PointXYZI>::Ptr angle_valid_pt2(new pcl::PointCloud<pcl::PointXYZI>());

      pcl::PointXYZI angle_lim_pt;
      angle_lim_pt.x = convex_hull_pt[min_index].x;
      angle_lim_pt.y = convex_hull_pt[min_index].y;
      angle_lim_pt.z = 0;
      angle_lim_pt.intensity = 1;
      angle_valid_pt2->push_back(angle_lim_pt);

      angle_lim_pt.x = convex_hull_pt[max_index].x;
      angle_lim_pt.y = convex_hull_pt[max_index].y;
      angle_lim_pt.z = 0;
      angle_lim_pt.intensity = 2;
      angle_valid_pt2->push_back(angle_lim_pt);

      // visualize the lim convex hull points
      sensor_msgs::PointCloud2 angle_valid_pt2_ros;
      pcl::toROSMsg(*angle_valid_pt2, angle_valid_pt2_ros);
      angle_valid_pt2_ros.header.frame_id = frame_id_;
      angle_valid_pt2_ros.header.stamp = ros::Time::now();
      pub_angle_valid_pt2_.publish(angle_valid_pt2_ros);
    }

    // Time the whole process
    const auto end_time = std::chrono::steady_clock::now();
    const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\033[1;36m [KITTI3D Loader] took " << elapsed_time.count() << " milliseconds" << "\033[0m" << std::endl;

    r.sleep();
  }

  return 0;
}