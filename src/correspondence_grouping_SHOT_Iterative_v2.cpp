//CORRESPONDENCE GROUPING FRAMEWORK VERSION 2.0
//NEEDS SCENE PCD FILES AND CORRESPONDING ANNOTATION XML FILES IN THE SAME DIRECTORY
//EXTRACTS OBJECTS AND FINDS CORRESPONDENCES IN ALL THE SCENES

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

#include <ios>
#include <sstream>
#include <boost/property_tree/xml_parser.hpp>

#include <pcl/point_types.h>
#include <boost/property_tree/ptree.hpp>
#include <pcl/filters/passthrough.h>


#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>

#include <string>

namespace fs = boost::filesystem;
using namespace boost::property_tree;

static const std::string TAG_SCENARIO = "scenario";
static const std::string TAG_NAME = "name";
static const std::string TAG_OBJECT = "object";
static const std::string TAG_NOBJECTS = "numberOfObjects";
static const std::string TAG_ALLOBJECTS = "allObjects";
static const std::string TAG_POSE = "pose";
static const std::string TAG_DIMENSIONS = "dimensions";
static const std::string TAG_LENGTH = "length";
static const std::string TAG_WIDTH = "width";
static const std::string TAG_HEIGHT = "height";
static const std::string TAG_COLOR = "color";
static const std::string TAG_INDICES = "indices";

pcl::PCDReader reader;
pcl::PCDWriter writer;

class object
{
public:
    std::string name;
    std::string color;
    pcl::PointIndices indices;  
};

std::vector<object> _objectList;

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
float model_ss_ (0.01f);
float scene_ss_ (0.03f);
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);

pcl::PointCloud<PointType>::Ptr model_cloud (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr scene_cloud (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());

int score[10][10] = {};
int s_file_count = 0;
int m_file_count = 0;

std::string model_filename;


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


fs::path
parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  //Directory path
  	fs::path folder_path( fs::initial_path<fs::path>());

 	if ( argc == 2 )
 	{
    	folder_path = fs::system_complete( fs::path( argv[1]));
    	std::cout << folder_path.string() << std::endl;
	}

	else
	{
	    showHelp (argv[0]);
	    exit (-1);
	}

	if (!fs::exists(folder_path))
	{
	    std::cout << "\nNot found: " << folder_path.string() << std::endl;
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
  return folder_path.string();
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


std::vector<std::string> 
pathIteration (fs::path folder)
{
	std::string fileN;
	std::vector<std::string> filenames;
	fs::directory_iterator it_s(folder);
    fs::directory_iterator endit_s;
    std::cout << "Files found:" << std::endl;
    while(it_s != endit_s) //for every scene
    	{  
    		if(fs::is_regular_file(*it_s) && it_s->path().extension() == ".pcd") 
      		{ 
        		fileN = folder.string() + it_s->path().filename().string();
        		fileN = fileN.erase(fileN.find_last_of("."), 4);
        		std::cout << fileN << std::endl;
        		filenames.push_back(fileN);
        	}

        	++it_s;
        }

    return filenames;
}


void
displayScore()
{ 
  std::cout<<"\t";
  //std::cout << m_file_count << "      " << s_file_count;

  for (int m = -1; m < m_file_count ; m++)
  {
    for (int s = -1; s < s_file_count; s++)
    {
        if (m < 0)
        {
          if (s>-1)
          std::cout << "scene_" << s << "\t";
        }
        else
        {
          if (s<0)
            std::cout << "model_" << m << "\t";
          else
            std::cout << score[m][s] << "\t";
        }
    }
    
    std::cout<<std::endl;
  }

}


void parseObject(ptree &parent){
    object newObject;
    // Name and color
    newObject.name = parent.get<std::string>(TAG_NAME);
    newObject.color = parent.get<std::string>(TAG_COLOR);

    // Get indices
    std::istringstream str(parent.get<std::string>(TAG_INDICES));
    newObject.indices.indices.clear();
    int i;
    while(str >> i){
        newObject.indices.indices.push_back(i);
    }
    _objectList.push_back(newObject);
}


void importObjectsInformation(const char* xmlFile)
{
    ptree root;
    read_xml(xmlFile, root);
    //std::string scenarioType = root.get<std::string>(TAG_SCENARIO + "." + "type");
    //ptree& tableDimensions = root.get_child(TAG_SCENARIO + "." + TAG_DIMENSIONS);
   
    ptree& allObjects = root.get_child(TAG_SCENARIO + "." + TAG_ALLOBJECTS);

    ptree::iterator it = allObjects.begin();
    it++;
    for(; it != allObjects.end(); it++){
        parseObject(it->second);
    }

    //return scenarioType;
}


void displayObjects()
{
	for (std::vector<object>::const_iterator it = _objectList.begin (); it != _objectList.end (); ++it)
	{
    std::cout << it->name << std::endl;
    }
}

void createDir(std::string folder)
{
	fs::path f_path = fs::system_complete( fs::path(folder));

	if (!fs::exists(f_path))
	{
		const int dir_err = mkdir(folder.c_str(), ACCESSPERMS);
		if (dir_err == -1)
		{
    		printf("Error creating directory!");
    		exit(0);
		}	
	}
}

void extractPCD(std::vector<object> objList, std::string folder, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_)
{	
	for(int i = 0; i < objList.size(); i++)
  	{
  		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBA>);
    	for (std::vector<int>::const_iterator pit = objList[i].indices.indices.begin (); pit != objList[i].indices.indices.end (); ++pit)
    	//for(int j = 0; j < objList[i].indices.indices.size(); j++)
      		cloud_cluster->points.push_back (scene_->points[*pit]); //*
    
    	cloud_cluster->width = cloud_cluster->points.size ();
    	cloud_cluster->height = 1;
    	cloud_cluster->is_dense = true;
    	std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    	std::stringstream ss;
    	ss << folder << "/"<< objList[i].name << ".pcd";
    	writer.write<pcl::PointXYZRGBA> (ss.str (), *cloud_cluster, false); //*
  	}
}


int
correspondenceGrouping (std::string model, std::string scene)
{ 
//  Load clouds
	if (pcl::io::loadPCDFile (model, *model_cloud) < 0)
  	{
   		std::cout << "Error loading model cloud." << std::endl;
    	return (-1);
  	}

  	if (pcl::io::loadPCDFile (scene, *scene_cloud) < 0)
  	{
   		std::cout << "Error loading scene cloud." << std::endl;
    	return (-1);
 	 }

//  Compute Cloud Resolution

  computeCloudResolution(model_cloud);

//  Compute Normals

  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model_cloud);
  norm_est.compute (*model_normals);

  norm_est.setInputCloud (scene_cloud);
  norm_est.compute (*scene_normals);

//  Downsample Clouds to Extract keypoints

  pcl::PointCloud<int> sampled_indices;

  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud (model_cloud);
  uniform_sampling.setRadiusSearch (model_ss_);
  uniform_sampling.compute (sampled_indices);
  pcl::copyPointCloud (*model_cloud, sampled_indices.points, *model_keypoints);
  std::cout << "Model total points: " << model_cloud->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

  uniform_sampling.setInputCloud (scene_cloud);
  uniform_sampling.setRadiusSearch (scene_ss_);
  uniform_sampling.compute (sampled_indices);
  pcl::copyPointCloud (*scene_cloud, sampled_indices.points, *scene_keypoints);
  std::cout << "Scene total points: " << scene_cloud->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;

//  Compute Descriptor for keypoints

  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);

  descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model_cloud);
  descr_est.compute (*model_descriptors);

  descr_est.setInputCloud (scene_keypoints);
  descr_est.setInputNormals (scene_normals);
  descr_est.setSearchSurface (scene_cloud);
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
    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
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
    rf_est.setSearchSurface (model_cloud);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene_cloud);
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
  
 return rototranslations.size();
}


void
CorrespondenceIteration(std::vector<std::string> fileNames, fs::path rootFolder)
{	
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGBA>());
	
	std::ofstream myfile;
	std::string resultFile = rootFolder.string() + "/Result.txt";
	myfile.open (resultFile.c_str());

	for (std::vector<std::string>::iterator it = fileNames.begin(); it != fileNames.end(); ++it)
	{
		std::string pcdFile = *it + ".pcd";
		std::string xmlFile = *it + ".xml";

		if (reader.read (pcdFile, *scene) < 0)
 		{
    		std::cout << "Error loading scene cloud." << std::endl;
    		exit (0);
  		}

  		importObjectsInformation(xmlFile.c_str());
 		//displayObjects();
 		createDir(*it);
 		extractPCD(_objectList, *it, scene);
 		_objectList.clear();

 		fs::path models_path( fs::initial_path<fs::path>());
 		models_path = fs::system_complete(fs::path(*it));

 		fs::directory_iterator it_s(rootFolder);
    	fs::directory_iterator endit_s;

    	myfile << "--------------------------------------------------------------------------------------" << std::endl;
    	myfile << *it << std::endl;
    	myfile << "--------------------------------------------------------------------------------------" << std::endl;


    	while(it_s != endit_s) //for every scene
    	{  
    		if(fs::is_regular_file(*it_s) && it_s->path().extension() == ".pcd") 
      		{ 
        		std::string scene_filename = rootFolder.string() + it_s->path().filename().string();
        		
		 		fs::directory_iterator it_m(models_path);
        		fs::directory_iterator endit_m;

        		while(it_m != endit_m) //for every model
        		{  
          			if(fs::is_regular_file(*it_m) && it_m->path().extension() == ".pcd") 
          			{ 
            			std::string model_filename = models_path.string() + "/" + it_m->path().filename().string();
            		// DO CORRESPONDENCE GROUPING
            			//std::cout << scene_filename << " <<<<<>>>>> " << model_filename << std::endl;
            			//std::cout << it_s->path().filename().string() << " <<<<<>>>>> " << it_m->path().filename().string() << std::endl;
            			//std::cout << "scene_" << j << "     model_" << i << "   Correspondence:  " <<  correspondenceGrouping(model_filename,scene_filename) <<std::endl;
            			myfile << it_s->path().filename().string() << " <<<<<>>>>> " << it_m->path().filename().string()<< "   :  " << correspondenceGrouping(model_filename,scene_filename) <<std::endl;
          			}

          			++it_m;
       			}

       		}

       		++it_s;
        }
	}

  	myfile.close();
}



int
main (int argc, char *argv[])
{
  fs::path folderName_ = parseCommandLine (argc, argv);
  std::vector<std::string> fileNames_ = pathIteration(folderName_);
  CorrespondenceIteration(fileNames_,folderName_);
  //displayScore();
}
