//OBJECT EXTRACTOR
//GIVEN A SCENE PCD FILES AND CORRESSPONDING ANNOTATION XML FILES IN A DIRECTORY
//EXTRACTS THE ANNOTATED OBJECTS FROM ALL SCENES

#include <fstream>
#include <iostream>
#include <string>
#include <ios>
#include <sstream>
#include <boost/property_tree/xml_parser.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/property_tree/ptree.hpp>
#include <pcl/filters/passthrough.h>

#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>

#include <string>

#include <stdio.h>
#include <time.h>
#include <sys/stat.h>

#include <cstddef>        // std::size_t

#include <signal.h>


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

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGBA>());
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

fs::path
parseCommandLine (int argc, char *argv[])
{
  //Directory path
    fs::path folder_path( fs::initial_path<fs::path>());

    if ( argc > 1 )
    {
        folder_path = fs::system_complete( fs::path( argv[1]));
        std::cout << folder_path.string() << std::endl;
    }

    else
    {
        std::cout << "Usage: objectExtractionIteration [pcd,xml folder]" << std::endl;
        exit (-1);
    }

    if (!fs::exists(folder_path))
    {
        std::cout << "\nNot found: " << folder_path.string() << std::endl;
        exit (-1);
    }

    return folder_path.string();
}


std::vector<std::string> 
pathIteration (fs::path folder)
{
    std::string fileN;
    std::vector<std::string> filenames;
    fs::directory_iterator it_s(folder);
    fs::directory_iterator endit_s;
    //std::cout << "Files found:" << std::endl;
    while(it_s != endit_s) //for every scene
        {  
            if(fs::is_regular_file(*it_s) && it_s->path().extension() == ".pcd") 
            { 
                fileN = folder.string() + "/" + it_s->path().filename().string();
                fileN = fileN.erase(fileN.find_last_of("."), 4);
                //std::cout << fileN << std::endl;
                filenames.push_back(fileN);
            }

            ++it_s;
        }

    return filenames;
}

void parseObject(ptree &parent)
{
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

    ptree& allObjects = root.get_child(TAG_SCENARIO + "." + TAG_ALLOBJECTS);

    ptree::iterator it = allObjects.begin();
    it++;
    for(; it != allObjects.end(); it++)
    {
     parseObject(it->second);
    }
}


void displayObjects()
{
    for (std::vector<object>::const_iterator it = _objectList.begin (); it != _objectList.end (); ++it)
    {
    std::cout << it->name << std::endl;
    }
}

std::string createDir()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[20];

  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tstruct);
  char dir[30] = "";
  strcat(dir,"Objects_");
  strcat(dir,buf);
  
  const int dir_err = mkdir(dir, ACCESSPERMS);

  if (dir_err == -1)
  {
   std::cout << "Error creating directory!" << std::endl ;
   exit(1);
  }

  std::string str(dir);
  return str;
}

   

void extractPCD(std::vector<object> objList, std::string folder, std::string file, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_)
{   
    for(int i = 0; i < objList.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBA>);
        for (std::vector<int>::const_iterator pit = objList[i].indices.indices.begin (); pit != objList[i].indices.indices.end (); ++pit)
      
        cloud_cluster->points.push_back (scene_->points[*pit]);
    
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        //std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
        std::stringstream ss;
        ss << folder << "/" << objList[i].name << "-" << file << ".pcd";
        //std::cout<< ss.str ();
        writer.write<pcl::PointXYZRGBA> (ss.str (), *cloud_cluster, false); //*
    }
}


void
ExtractionIteration(std::vector<std::string> fileNamesV, fs::path rootFolder)
{   
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGBA>());
    std::ofstream myfile;
    std::string folder = createDir();
    int i =0;
  //Extract all the objects from all the pcd in the root directory
    std::cout<< "No. of files = " << fileNamesV.size() << std::endl;

    for (std::vector<std::string>::iterator it = fileNamesV.begin(); it != fileNamesV.end(); ++it)
    {   
        std::string pcdFile = *it + ".pcd";
        std::string xmlFile = *it + ".xml";

        if (reader.read (pcdFile, *scene) < 0)
        {
        std::cout << "Error loading scene cloud from : " << pcdFile << std::endl;
        exit (0);
        }

        if(!fs::exists(xmlFile))
        {
        std::cout<<"Corresponding .xml file not found."<<std::endl;
        exit(0);
        }

        importObjectsInformation(xmlFile.c_str());
        //displayObjects();
        std::size_t found = it->find_last_of("/\\");
        std::string filNam = it->substr(found+1);
        extractPCD(_objectList, folder, filNam,  scene); //Extract object clusters as .pcd files into the created directory
        _objectList.clear();


    }
}

int
main (int argc, char *argv[])
{
    fs::path folderName_ = parseCommandLine (argc, argv);
    std::vector<std::string> fileNames_ = pathIteration(folderName_);
    ExtractionIteration(fileNames_,folderName_);
}
