//PCD SIZE FINDER
//FIND THE DATA POINTS IN A PCD FILE

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>

int 
main (int argc, char** argv)
{
  // Read in the cloud data
  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  reader.read (argv[1], *cloud);
  std::cout << "PointCloud has: " << cloud->points.size () << " data points." << std::endl; //*
  return (0);
}
