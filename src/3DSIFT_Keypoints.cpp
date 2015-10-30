#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

using namespace pcl;

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;

float cloud_ss_ (0.01f);


int
main (int argc, char *argv[])
{

  if (argc < 2)
  {
    std::cout << "Usage: 3DSIFT_Keypoints [pcd_file]" << std::endl;
    exit(0);
  }

  pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType> ());

  if (pcl::io::loadPCDFile (argv[1], *cloud) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    exit(0);
  }

//KEYPOINT DEFINITION


  pcl::PointCloud<pcl::PointWithScale>::Ptr cloud_keypoints;

  pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift_detect;

  // Use a FLANN-based KdTree to perform neighborhood searches
  sift_detect.setSearchMethod (pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZRGB>));
  // Set the detection parameters
  sift_detector.setScales (0.02, 3, 3);
  sift_detector.setMinimumContrast (10.0);
  // Set the input
  sift_detector.setInputCloud (cloud); 
  sift_detector.compute (cloud_keypoints);


  //pcl::UniformSampling<PointType> uniform_sampling;
  //uniform_sampling.setInputCloud (cloud);
  //uniform_sampling.setRadiusSearch (cloud_ss_);
  //uniform_sampling.compute (kp_indices);
  //pcl::copyPointCloud (*cloud, kp_indices.points, *cloud_keypoints);
  std::cout << "Scene total points: " << cloud->size () << std::endl;
  std::cout << "Selected Keypoints: " << cloud_keypoints->size() << std::endl;


  pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
  viewer.addPointCloud (cloud, "scene_cloud");

  pcl::visualization::PointCloudColorHandlerCustom<PointType> cloud_keypoints_color_handler (cloud_keypoints, 0, 0, 255);
  viewer.addPointCloud (cloud_keypoints, cloud_keypoints_color_handler, "cloud_keypoints");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud_keypoints");


  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }

  return (0);
}
