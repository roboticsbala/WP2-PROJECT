#include <opencv/cv.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "pcl/io/pcd_io.h"
#include <Eigen/Core>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/console/parse.h>

//using namespace pcl;
using namespace std;
using namespace cv;

bool visualize = false;
bool ransac = false;
bool icp = false;
bool tfc = false;

void
downsample (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, float leaf_size,
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &downsampled_out)
{
  //voxel grid filters
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid;
  vox_grid.setSaveLeafLayout (true);
  vox_grid.setLeafSize (leaf_size, leaf_size, leaf_size);
  vox_grid.setInputCloud (points);
  vox_grid.filter (*tmp_ptr);
  //added passthrough filters
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_ptr3(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud (tmp_ptr);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 3.0);
  pass.filter (*tmp_ptr3);
  pcl::PassThrough<pcl::PointXYZRGB> pass2;
  pass2.setInputCloud (tmp_ptr3);
  pass2.setFilterFieldName ("x");
  pass2.setFilterLimits (-2.0, 1.0);
  pass2.filter (*downsampled_out);

}

void
compute_surface_normals (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, float normal_radius,
                         pcl::PointCloud<pcl::Normal>::Ptr &normals_out)
{
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> norm_est;

  // Use a FLANN-based KdTree to perform neighborhood searches
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());

  norm_est.setSearchMethod (tree);

  // Specify the size of the local neighborhood to use when computing the surface normals
  norm_est.setRadiusSearch (normal_radius);

  // Set the input points
  norm_est.setInputCloud (points);

  // Estimate the surface normals and store the result in "normals_out"
  norm_est.compute (*normals_out);
}


void
compute_PFH_features (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, 
                      pcl::PointCloud<pcl::Normal>::Ptr &normals, 
                      float feature_radius,
                      pcl::PointCloud<pcl::PFHSignature125>::Ptr &descriptors_out)
{
  // Create a PFHEstimation object
  pcl::PFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHSignature125> pfh_est;

  // Set it to use a FLANN-based KdTree to perform its neighborhood searches
  pcl::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::KdTree<pcl::PointXYZRGB>)
  pfh_est.setSearchMethod (tree);

  // Specify the radius of the PFH feature
  pfh_est.setRadiusSearch (feature_radius);

  // Set the input points and surface normals
  pfh_est.setInputCloud (points);  
  pfh_est.setInputNormals (normals);  

  // Compute the features
  pfh_est.compute (*descriptors_out);
  
}
// Find indices for keypoints in the pointcloud
void getIndex(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &keypoints, vector<int> &index){
  pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
  kdtree.setInputCloud (points);
  int K=1;
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for(size_t i=0; i<keypoints->points.size(); i++){
    kdtree.nearestKSearch (keypoints->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance);
    index[i] = pointIdxNKNSearch[0];
  }
}

void
detect_keypoints (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points,
                  float min_scale, int nr_octaves, int nr_scales_per_octave, float min_contrast,
                  pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints_out)
{
  pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift_detect;

  // Use a FLANN-based KdTree to perform neighborhood searches
  sift_detect.setSearchMethod (pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZRGB>));

  // Set the detection parameters
  sift_detect.setScales (min_scale, nr_octaves, nr_scales_per_octave);
  sift_detect.setMinimumContrast (min_contrast);

  // Set the input
  sift_detect.setInputCloud (points);

  // Detect the keypoints and store them in "keypoints_out"
  sift_detect.compute (*keypoints_out);
}

void
compute_PFH_features_at_keypoints (pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, 
                                   pcl::PointCloud<pcl::Normal>::Ptr &normals, 
                                   pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints, float feature_radius,
                                   pcl::PointCloud<pcl::PFHSignature125>::Ptr &descriptors_out, vector<int> &indices)
{
  // Create a PFHEstimation object
  pcl::PFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PFHSignature125> pfh_est;

  // Set it to use a FLANN-based KdTree to perform its neighborhood searches
  pfh_est.setSearchMethod (pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr (new pcl::KdTreeFLANN<pcl::PointXYZRGB>));

  // Specify the radius of the PFH feature
  pfh_est.setRadiusSearch (feature_radius);

  /* This is a little bit messy: since our keypoint detection returns PointWithScale points, but we want to
   * use them as an input to our PFH estimation, which expects clouds of PointXYZRGB points.  To get around this,
   * we'll use copyPointCloud to convert "keypoints" (a cloud of type PointCloud<PointWithScale>) to 
   * "keypoints_xyzrgb" (a cloud of type PointCloud<PointXYZRGB>).  Note that the original cloud doesn't have any RGB 
   * values, so when we copy from PointWithScale to PointXYZRGB, the new r,g,b fields will all be zero.
   */
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_xyzrgb (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud (*keypoints, *keypoints_xyzrgb);

  // Use all of the points for analyzing the local structure of the cloud
  pfh_est.setSearchSurface (points);  
  pfh_est.setInputNormals (normals);  

  // But only compute features at the keypoints
  pfh_est.setInputCloud (keypoints_xyzrgb);

  // Compute the features
  pfh_est.compute (*descriptors_out);

  getIndex(points, keypoints_xyzrgb, indices);

}

void
find_feature_correspondences (pcl::PointCloud<pcl::PFHSignature125>::Ptr &source_descriptors,
                              pcl::PointCloud<pcl::PFHSignature125>::Ptr &target_descriptors,
                              std::vector<int> &correspondences_out, std::vector<float> &correspondence_scores_out)
{
  // Resize the output vector
  correspondences_out.resize (source_descriptors->size ());
  correspondence_scores_out.resize (source_descriptors->size ());

  // Use a KdTree to search for the nearest matches in feature space
  pcl::KdTreeFLANN<pcl::PFHSignature125> descriptor_kdtree;
  descriptor_kdtree.setInputCloud (target_descriptors);

  // Find the index of the best match for each keypoint, and store it in "correspondences_out"
  const int k = 1;
  std::vector<int> k_indices (k);
  std::vector<float> k_squared_distances (k);
  for (size_t i = 0; i < source_descriptors->size (); ++i)
  {
    descriptor_kdtree.nearestKSearch (*source_descriptors, i, k, k_indices, k_squared_distances);
    correspondences_out[i] = k_indices[0];
    correspondence_scores_out[i] = k_squared_distances[0];
  }
}

/* To investigate on if the closest match to the target descriptor is also the previously mathced source */
void experiment_correspondences (pcl::PointCloud<pcl::PFHSignature125>::Ptr &source_descriptors,
                              pcl::PointCloud<pcl::PFHSignature125>::Ptr &target_descriptors,
                              vector<int> &correct){
  vector<int> corr1(source_descriptors->size ());
  vector<int> corr2(target_descriptors->size ());
  correct.resize(source_descriptors->size());
  // Use a KdTree to search for the nearest matches in feature space
  pcl::KdTreeFLANN<pcl::PFHSignature125> descriptor_kdtree;
  descriptor_kdtree.setInputCloud (target_descriptors);

  // Find the index of the best match for each keypoint, and store it in "correspondences_out"
  const int k = 1;
  std::vector<int> k_indices (k);
  std::vector<float> k_squared_distances (k);
  for (size_t i = 0; i < source_descriptors->size (); ++i)
  {
    descriptor_kdtree.nearestKSearch (*source_descriptors, i, k, k_indices, k_squared_distances);
    corr1[i] = k_indices[0];
  }

  // Use a KdTree to search for the nearest matches in feature space
  pcl::KdTreeFLANN<pcl::PFHSignature125> descriptor_kdtree2;
  descriptor_kdtree2.setInputCloud (source_descriptors);

  // Find the index of the best match for each keypoint, and store it in "correspondences_out"
  std::vector<int> k_indices2 (k);
  std::vector<float> k_squared_distances2 (k);
  for (size_t i = 0; i < target_descriptors->size (); ++i)
  {
    descriptor_kdtree2.nearestKSearch (*target_descriptors, i, k, k_indices2, k_squared_distances2);
    corr2[i] = k_indices2[0];
  }
  int count = 0;
  for(size_t i=0; i<source_descriptors->points.size(); i++){
    if(abs(corr1[corr2[i]])!=i){ count++; correct[i]=0;}
    else{
      correct[i]=1;
    }
  }
  cout<<"Not matched correspondences : "<<count<<endl;
}

void visualize_correspondences (const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points1,
                                const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints1,
                                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr points2,
                                const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints2,
                                const std::vector<int> &correspondences,
                                const std::vector<float> &correspondence_scores, const std::vector<int> index1, const std::vector<int> index2)
{
  // We want to visualize two clouds side-by-side, so do to this, we'll make copies of the clouds and transform them
  // by shifting one to the left and the other to the right.  Then we'll draw lines between the corresponding points

  // Create some new point clouds to hold our transformed data
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_left (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_left (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_right (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_right (new pcl::PointCloud<pcl::PointWithScale>);

  // Shift the first clouds' points to the left
  const Eigen::Vector3f translate (2.0, 0.0, 0.0);
  const Eigen::Quaternionf no_rotation (0, 0, 0, 0);
  pcl::transformPointCloud (*points1, *points_left, -translate, no_rotation);
  pcl::transformPointCloud (*keypoints1, *keypoints_left, -translate, no_rotation);

  // Shift the second clouds' points to the right
  pcl::transformPointCloud (*points2, *points_right, translate, no_rotation);
  pcl::transformPointCloud (*keypoints2, *keypoints_right, translate, no_rotation);

  // Add the clouds to the vizualizer
  pcl::visualization::PCLVisualizer viz;
  viz.addPointCloud (points_left, "points_left");
  viz.addPointCloud (points_right, "points_right");

  // Compute the median correspondence score
  std::vector<float> temp (correspondence_scores);
  std::sort (temp.begin (), temp.end ());
  float median_score = temp[temp.size ()/2];

  // Draw lines between the best corresponding points
  for (size_t i = 0; i < keypoints_left->size (); ++i)
  {
   //   if (correct[i]==0) continue;
    if (correspondence_scores[i] > median_score)
    {
      continue; // Don't draw weak correspondences
    }

    // Get the pair of points
    const pcl::PointWithScale & p_left = keypoints_left->points[i];
    const pcl::PointWithScale & p_right = keypoints_right->points[correspondences[i]];
   if(abs(p_left.y-p_right.y)>0.2) continue;

    // Generate a random (bright) color
    double r = (rand() % 100);
    double g = (rand() % 100);
    double b = (rand() % 100);
    double max_channel = std::max (r, std::max (g, b));
    r /= max_channel;
    g /= max_channel;
    b /= max_channel;

    // Generate a unique string for each line
    std::stringstream ss ("line");
    ss << i;

    // Draw the line
    viz.addLine (p_left, p_right, r, g, b, ss.str ());
  }

  // Give control over to the visualizer
  viz.spin ();
}

int transform_demo (const char * filename1, const char *filename2)
{
  // Create some new point clouds to hold our data
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr points1 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled1 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::Normal>::Ptr normals1 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints1 (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors1 (new pcl::PointCloud<pcl::PFHSignature125>);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr points2 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled2 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::Normal>::Ptr normals2 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints2 (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors2 (new pcl::PointCloud<pcl::PFHSignature125>);

  // Load the pair of point clouds
  if(pcl::io::loadPCDFile (filename1, *points1)==-1)
  {
    PCL_ERROR ("Couldn't read first file! \n");
    return (-1);
  }
  if(pcl::io::loadPCDFile (filename2, *points2)==-1) 
  {
    PCL_ERROR ("Couldn't read second input file! \n");
    return (-1);
  }
  // Downsample the cloud
  const float voxel_grid_leaf_size = 0.01;
  downsample (points1, voxel_grid_leaf_size, downsampled1);
  downsample (points2, voxel_grid_leaf_size, downsampled2);  
  // save downsampled input pointclouds for debug purposes
  /* pcl::io::savePCDFileASCII ("downsampled1.pcd", *downsampled1);
     pcl::io::savePCDFileASCII ("downsampled2.pcd", *downsampled2); */
  // Compute surface normals
  const float normal_radius = 0.03;
  compute_surface_normals (downsampled1, normal_radius, normals1);
  compute_surface_normals (downsampled2, normal_radius, normals2);
  // Compute keypoints
  const float min_scale = 0.01;
  const int nr_octaves = 3;
  const int nr_octaves_per_scale = 3;
  const float min_contrast = 10.0;
  detect_keypoints (downsampled1, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints1);
  detect_keypoints (downsampled2, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints2);
  // Compute PFH features
  const float feature_radius = 0.08; 
  vector<int> indices1(keypoints1->points.size());
  compute_PFH_features_at_keypoints (downsampled1, normals1, keypoints1, feature_radius, descriptors1, indices1);
  vector<int> indices2(keypoints2->points.size());
  compute_PFH_features_at_keypoints (downsampled2, normals2, keypoints2, feature_radius, descriptors2, indices2);
  // Find feature correspondences
  std::vector<int> correspondences;
  std::vector<float> correspondence_scores;
  find_feature_correspondences (descriptors1, descriptors2, correspondences, correspondence_scores);
  /* to experiment on the correspondences */
  /*  vector<int> correct;
      experiment_correspondences(descriptors1, descriptors2, correct);*/
  // Print out ( number of keypoints / number of points )
  std::cout << "First cloud: Found " << keypoints1->size () << " keypoints "
            << "out of " << downsampled1->size () << " total points." << std::endl;
  std::cout << "Second cloud: Found " << keypoints2->size () << " keypoints "
            << "out of " << downsampled2->size () << " total points." << std::endl;

  if(tfc){
    /*********TFC**********/
    /* Compute the transformation between the clouds using Transformation from Correspondences class*/
    pcl::TransformationFromCorrespondences tfc;
    tfc.reset();
    // sort correspondences based on scores and pass them to transformation from correspondences.
    std::vector<int> sorted_scores;
    cv::sortIdx(correspondence_scores,sorted_scores, 2);
    // Compute the median correspondence score
    std::vector<float> temp (correspondence_scores);
    std::sort (temp.begin (), temp.end ());
    float median_score = temp[temp.size ()/2];
    vector<int> fidx;
    vector<int> fidxt;
    Eigen::Vector3f source_position(0, 0, 0); 	Eigen::Vector3f target_position(0, 0, 0); 
   /* Breifly Transform from correspondences incrementally computes the covariance and the two means of the sets
     (the "point set" and the "correspondences set") as the points come in, then it gets the rotation from the SVD of the cov and the translation from the means.*/
    for(size_t i=0; i<correspondence_scores.size(); i++ ){
 	int index = sorted_scores[i];
	if(median_score>=correspondence_scores[index]){
		source_position[0]=keypoints1->points[index].x; source_position[1]=keypoints1->points[index].y; source_position[2]=keypoints1->points[index].z;
		target_position[0]=keypoints2->points[correspondences[index]].x; target_position[1]=keypoints2->points[correspondences[index]].y; 
                           target_position[2]=keypoints2->points[correspondences[index]].z;
		// Assuming the camera/kinect does not move in along z axis.
		if(abs(source_position[1]-target_position[1])>0.2) continue;
		//if(correct[i]!=1) continue;
		tfc.add(source_position,target_position, correspondence_scores[index]);
		fidx.push_back(indices1[index]);
		fidxt.push_back(indices2[correspondences[index]]);
	}
    }
    // Save the transformation from TFC
    std::ofstream out("transform_tfc.txt");
    // cout<<"Number of samples: "<< tfc.getNoOfSamples() <<endl;
    Eigen::Affine3f Tr;Tr = tfc.getTransformation();
    cout<<"TFC transformation: "<<endl;
    for(int i=0; i<4; i++){
      for(int j=0; j<4; j++){
	out<<" "<<Tr(i,j);
        cout<<Tr(i,j)<<"\t";
      }
      cout<<endl;
      out<<endl;
    }
    out.close();
    // Save the transformed cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud (*downsampled1, *cloud_in_ptr, tfc.getTransformation());
    pcl::io::savePCDFileASCII ("cloud_after_tfc.pcd", *cloud_in_ptr);
  }
  /*********ICP**********/
  if(icp){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    icp.setInputCloud(downsampled1);
    icp.setInputTarget(downsampled2);
    icp.setMaximumIterations (50);
    icp.align(*cloud_out);
    std::cout << "ICP has converged = " << icp.hasConverged() <<endl;
    // Save the transformed cloud
    pcl::io::savePCDFileASCII ("cloud_after_icp.pcd", *cloud_out);
    // Save the transformation
    std::ofstream out2("transform_icp.txt");
    Eigen::Affine3f Tr_icp;Tr_icp = icp.getFinalTransformation ();
    cout<<"ICP transformation: "<<endl;
    for(int i=0; i<4; i++){
      for(int j=0; j<4; j++){
	out2<<" "<<Tr_icp(i,j);
        cout<<Tr_icp(i,j)<<"\t";
      }
      cout<<endl;
      out2<<endl;
    }
    out2.close();
  }
  /********RANSAC********/
  if(ransac){
    pcl::SampleConsensusModelRegistration< PointXYZRGB >::Ptr model_r(new pcl::SampleConsensusModelRegistration< PointXYZRGB > (downsampled1, indices1) );
    vector<int> target_indices(correspondences.size());
    for(size_t i=0; i<correspondences.size() ;i++)
       target_indices[i] = indices2[correspondences[i]];
    Eigen::VectorXf coeff;
    model_r->setInputTarget (downsampled2, target_indices);
    pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_r, 0.05);
    ransac.computeModel(0);
    ransac.getModelCoefficients(coeff);
    Eigen::Matrix4f transform_ransac;
    cout<<"RANSAC transformation: "<<endl;
    for(size_t i=0;i<16;i++){
       transform_ransac(i/4,i%4)=coeff[i];
       cout<<coeff[i]<<"\t";
       if(i%4==0) cout<<endl;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud (*downsampled1, *transformed_cloud2, transform_ransac);
    pcl::io::savePCDFileASCII ("cloud_after_ransac.pcd", *transformed_cloud2);
    // Always run ICP after ransac
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out2(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp2;
    icp2.setInputCloud(transformed_cloud2);
    icp2.setInputTarget(downsampled2);
    icp2.setMaximumIterations (50);
    icp2.align(*cloud_out2);
    std::cout << "ICP has converged = " << icp2.hasConverged() <<endl;
    // Save the transformed cloud
    pcl::io::savePCDFileASCII ("cloud_after_ransac_icp.pcd", *cloud_out2);
    Eigen::Matrix4f Tr_ransac_icp=icp2.getFinalTransformation ();
    Tr_ransac_icp=transform_ransac*Tr_ransac_icp;
    std::ofstream out3("transform_ransac_icp.txt");
    out3<<  Tr_ransac_icp <<endl;
    out3.close();
    cout<<"RANSAC + ICP transformation: "<<endl;
    for(unsigned int i=0; i<4; i++){
      for(unsigned int j=0; j<4; j++)
        cout<<Tr_ransac_icp(i,j)<<"\t";
      cout<<endl; 
    }
  }
  // Visualize the two point clouds and their feature correspondences
  if(visualize){
    //  visualize_correspondences (points1, keypoints1, points2, keypoints2, correspondences, correspondence_scores, indices1, indices2, correct);
    visualize_correspondences (downsampled1, keypoints1, downsampled2, keypoints2, correspondences, correspondence_scores, indices1, indices2);
  }
  return 0;
}

/*** Help ***/
void 
printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" <filename1.pcd> <filename2.pcd> [options]\n\n"
            << "Options:\n"
            << "-------------------------------------------\n"
            << "-v           Visualize the correspondences\n"
            << "-ransac      Compute the transformation between the clouds using ransac on correspondences (Note: This always proceeds with an ICP)\n"
            << "-icp         Compute the transformation between the clouds using ICP\n"
            << "-tfc         Compute the transformation between the clouds using transformation From Correspondences\n"
            << "-h           this help\n"
            << "\n\n";
}

int main (int argc, char ** argv)
{
  // --------------------------------------
  // -----Parse Command Line Arguments-----
  // --------------------------------------
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    printUsage (argv[0]);
    return 0;
  }
  if (pcl::console::find_argument (argc, argv, "-v") >= 0)
  {
    visualize = true;
  } 
  if (pcl::console::find_argument (argc, argv, "-ransac") >= 0)
  {
    ransac = true;
  }   
  if (pcl::console::find_argument (argc, argv, "-icp") >= 0)
  {
    icp = true;
  } 
  if (pcl::console::find_argument (argc, argv, "-tfc") >= 0)
  {
    tfc = true;
  } 
  return (transform_demo (argv[1], argv[2]));

}
