//CORRESPONDENCE GROUPING OBJ-OBJ FRAMEWORK VERSION 1.0
//NEEDS OBJECT MODEL CATEGORY A, B IN DIFFERENT DIRECTORIES

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>

#include <string>
#include <signal.h>

namespace fs = boost::filesystem;

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);

float model_ss_ (0.015f);
float scene_ss_ (0.015f);

float rf_rad_ (0.06f);
float descr_rad_ (0.08f);

float cg_size_ (0.035f);
float cg_thresh_ (6.0f);

float kd_thresh_ (0.24f);

pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());

std::string model_filename_;
std::string scene_filename_;

fs::path model_path( fs::initial_path<fs::path>());
fs::path scene_path( fs::initial_path<fs::path>());

int true_pos = 0;
int true_neg = 0;
int false_pos = 0;
int false_neg = 0;
int total_iter = 0;

std::ofstream myfile;

void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_path scene_path [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     -k:                     Show used keypoints." << std::endl;
  std::cout << "     -c:                     Show used correspondences." << std::endl;
  std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
  std::cout << "                             each radius given by that value." << std::endl;
  std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
  std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}

void
parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  //Model & scene directory path

  if ( argc > 2 )
  {
    model_path = fs::system_complete( fs::path( argv[1]));
    scene_path = fs::system_complete( fs::path( argv[2]));
    std::cout << model_path.string() << std::endl;
    std::cout << scene_path.string() << std::endl;
  }

  else
  {
    showHelp (argv[0]);
    exit (-1);
  }

  if (!fs::exists(model_path))
  {
    std::cout << "\nNot found: " << model_path.string() << std::endl;
    exit (-1);
  }

  if (!fs::exists(scene_path))
  {
    std::cout << "\nNot found: " << scene_path.string() << std::endl;
    exit (-1);
  }

//Program behavior
  
  if (pcl::console::find_switch (argc, argv, "-k"))
  {
    show_keypoints_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-c"))
  {
    show_correspondences_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-r"))
  {
    use_cloud_resolution_ = true;
  }

  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough_ = true;
    }else if (used_algorithm.compare ("GC") == 0)
    {
      use_hough_ = false;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }

//General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
  pcl::console::parse_argument (argc, argv, "--kd_thresh", kd_thresh_);
}

void
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  //  Set up resolution invariance
  //
  if (use_cloud_resolution_)
  {
    float resolution = static_cast<float>(res);
    if (resolution != 0.0f)
    {
      model_ss_   *= resolution;
      scene_ss_   *= resolution;
      rf_rad_     *= resolution;
      descr_rad_  *= resolution;
      cg_size_    *= resolution;
    }

    std::cout << "Model resolution:       " << resolution << std::endl;
    std::cout << "Model sampling size:    " << model_ss_ << std::endl;
    std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
    std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
    std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
    std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
  }
}

int
loadCloud ()
{
  if (pcl::io::loadPCDFile (model_filename_, *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    return (-1);
  }

  if (pcl::io::loadPCDFile (scene_filename_, *scene) < 0)
  {
    std::cout << "Error loading scene cloud." << std::endl;
    return (-1);
  }
}

std::vector<int>
correspondenceGroup ()
{
  
//  Load clouds
  loadCloud ();

//  Compute Cloud Resolution

  computeCloudResolution(model);

//  Compute Normals

  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model);
  norm_est.compute (*model_normals);

  norm_est.setInputCloud (scene);
  norm_est.compute (*scene_normals);

//  Downsample Clouds to Extract keypoints

  pcl::PointCloud<int> sampled_indices;

  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud (model);
  uniform_sampling.setRadiusSearch (model_ss_);
  uniform_sampling.compute (sampled_indices);
  pcl::copyPointCloud (*model, sampled_indices.points, *model_keypoints);
  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

  uniform_sampling.setInputCloud (scene);
  uniform_sampling.setRadiusSearch (scene_ss_);
  uniform_sampling.compute (sampled_indices);
  pcl::copyPointCloud (*scene, sampled_indices.points, *scene_keypoints);
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;

//  Compute Descriptor for keypoints

  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);

  descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  descr_est.compute (*model_descriptors);

  descr_est.setInputCloud (scene_keypoints);
  descr_est.setInputNormals (scene_normals);
  descr_est.setSearchSurface (scene);
  descr_est.compute (*scene_descriptors);

//  Find Model-Scene Correspondences with KdTree

  pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

  pcl::KdTreeFLANN<DescriptorType> match_search;
  match_search.setInputCloud (model_descriptors);

  //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  for (size_t i = 0; i < scene_descriptors->size (); ++i)
  {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);
    if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    if(found_neighs == 1 && neigh_sqr_dists[0] < kd_thresh_) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
    }
  }

  std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
  
//  Actual Clustering

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  std::vector<pcl::Correspondences> clustered_corrs;

//  Using Hough3D
  if (use_hough_)
  {
    //
    //  Compute (Keypoints) Reference Frames only for Hough
    //
    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (rf_rad_);

    rf_est.setInputCloud (model_keypoints);
    rf_est.setInputNormals (model_normals);
    rf_est.setSearchSurface (model);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene);
    rf_est.compute (*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_);
    clusterer.setHoughThreshold (cg_thresh_);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //clusterer.cluster (clustered_corrs);
    clusterer.recognize (rototranslations, clustered_corrs);
  }

  else // Using GeometricConsistency
  {
    pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
    gc_clusterer.setGCSize (cg_size_);
    gc_clusterer.setGCThreshold (cg_thresh_);

    gc_clusterer.setInputCloud (model_keypoints);
    gc_clusterer.setSceneCloud (scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

    //gc_clusterer.cluster (clustered_corrs);
    gc_clusterer.recognize (rototranslations, clustered_corrs);
  }

//  Output results

  std::cout << "Model instances found: " << rototranslations.size () << std::endl;

  /*for (size_t i = 0; i < rototranslations.size (); ++i)
  {
    std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
    std::cout << "      Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
    Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
  }*/
  std::vector<int> result;
  result.push_back (int(model_scene_corrs->size ()));
  result.push_back (rototranslations.size());
  return result;
}
 

void calculate_save()
{	
	myfile << std::endl << "Total iterations : "<< total_iter << std::endl;
	myfile << "True positive: "<< true_pos  << "\t" << "Rate:  " << (float)true_pos/total_iter << std::endl;
    myfile << "True negative: "<< true_neg << "\t" << "Rate:  " << (float)true_neg/total_iter << std::endl;
    myfile << "False positive: "<< false_pos << "\t" << "Rate:  " << (float)false_pos/total_iter << std::endl;
    myfile << "False negative: "<< false_neg << "\t" << "Rate:  " << (float)false_neg/total_iter << std::endl;

    myfile.close();
}

void
pathIteration()
{
  if (fs::is_directory(scene_path)&& fs::is_directory(model_path))
  { 
    
    const std::string ext = ".pcd";
   //Creating Iterators for Model and Scene directories

    fs::recursive_directory_iterator it_s(scene_path);
    fs::recursive_directory_iterator endit_s;

    // Looping over all models and scenes
    std::stringstream resultFile;
    resultFile << "Result_" << model_ss_ << "_" << rf_rad_ <<  "_" <<  descr_rad_ << "_" << cg_size_ << "_" << cg_thresh_ << "_" << kd_thresh_ << ".txt";
    myfile.open (resultFile.str().c_str());
    myfile << "Parameters: " << std::endl << "model_ss_ : " << model_ss_ << std::endl;
    myfile << "rf_rad_ : " << rf_rad_ << std::endl << "descr_rad_ : " << descr_rad_ << std::endl;
    myfile << "cg_size_ : " <<  cg_size_ << std::endl << "cg_thresh_ : " <<  cg_thresh_ << std::endl;
    myfile << "kd_thresh_ : " << kd_thresh_ << std::endl;
    
    while(it_s != endit_s) //for every scene
    {  
      if(fs::is_regular_file(*it_s) && it_s->path().extension() == ext) 
      {
        scene_filename_ = scene_path.string() + it_s->path().filename().string();
        fs::recursive_directory_iterator it_m(model_path);
        fs::recursive_directory_iterator endit_m;

        while(it_m != endit_m) //for every model
        {  
          if(fs::is_regular_file(*it_m) && it_m->path().extension() == ext) 
          { 
            model_filename_ = model_path.string() + it_m->path().filename().string();
            // DO CORRESPONDENCE GROUPING
            std::vector<int> res = correspondenceGroup();
            myfile << it_s->path().filename().string() << " <<<>>> " << it_m->path().filename().string() << " : " << res[0] << " -> " << res[1] << std::endl;
            std::size_t  s_idx = it_s->path().filename().string().find("-");
            std::size_t  m_idx = it_m->path().filename().string().find("-");
            if (s_idx!=std::string::npos || m_idx!=std::string::npos)
            {
               //std::cout<< s_idx << std::endl;
               //std::cout<< m_idx << std::endl;
        	}
        	else
        		std::cout<<"Not found." <<std::endl;

            std::cout<< it_s->path().filename().string().substr(0,s_idx) << std::endl;
            std::cout<< it_m->path().filename().string().substr(0,m_idx) << std::endl;

            if (it_s->path().filename().string().substr(0,s_idx) == it_m->path().filename().string().substr(0,m_idx) && res[1]  > 0)
          	{
          		true_pos ++;
          	}

          	if (it_s->path().filename().string().substr(0,s_idx) == it_m->path().filename().string().substr(0,m_idx) && res[1]  < 1)
          	{
          		false_neg++;
          	}

          	if (it_s->path().filename().string().substr(0,s_idx) != it_m->path().filename().string().substr(0,m_idx) && res[1]  < 1)
          	{
          		true_neg++;
          	}

          	if (it_s->path().filename().string().substr(0,s_idx) != it_m->path().filename().string().substr(0,m_idx) && res[1]  > 0)
          	{
          		false_pos++;
          	}

          	total_iter ++; 

          ++it_m;
        }
      }
      ++it_s;
    }

}
	calculate_save();
    
  }
}


void save_function(int sig)
{ // can be called asynchronously
  calculate_save();
  exit(0);
} 

int
main (int argc, char *argv[])
{
	signal(SIGINT, save_function); 

	parseCommandLine (argc, argv);
	pathIteration();
  	//displayScore();
}
