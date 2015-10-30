//OBJECT EXTRACTOR
//GIVEN A SCENE PCD AND CORRESSPONDING ANNOTATION XML FILE
//EXTRACTS THE ANNOTATED OBJECTS

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

using namespace boost::property_tree;

static const std::string TAG_SCENARIO = "scenario";
static const std::string TAG_NAME = "name";
static const std::string TAG_OBJECT = "object";
static const std::string TAG_NOBJECTS = "numberOfObjects";
static const std::string TAG_ALLOBJECTS = "allObjects";
static const std::string TAG_POSE = "pose";
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

std::string importObjectsInformation(char *xmlFile)
{
    ptree root;
    read_xml(xmlFile, root);

    ptree& allObjects = root.get_child(TAG_SCENARIO + "." + TAG_ALLOBJECTS);
    std::string scenarioType = root.get<std::string>(TAG_SCENARIO + "." + "type");
  
    ptree::iterator it = allObjects.begin();
    it++;
    for(; it != allObjects.end(); it++){
        parseObject(it->second);
    }

    return scenarioType;
}

void displayObjects()
{
	for (std::vector<object>::const_iterator it = _objectList.begin (); it != _objectList.end (); ++it)
	{
    std::cout << it->name << std::endl;
    }
}

std::string createDir(std::string filename)
{
	
	filename = filename.erase(filename.find_last_of("."), 4);
	const int dir_err = mkdir(filename.c_str(), ACCESSPERMS);
	if (dir_err == -1)
	{
    printf("Error creating directory!");
    exit(0);
	}	
	return filename;
}

void extractPCD(std::vector<object> objList, std::string folder)
{	
	for(int i = 0; i < objList.size(); i++)
  	{
  		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBA>);
    	for (std::vector<int>::const_iterator pit = objList[i].indices.indices.begin (); pit != objList[i].indices.indices.end (); ++pit)
    	//for(int j = 0; j < objList[i].indices.indices.size(); j++)
      		cloud_cluster->points.push_back (scene->points[*pit]); //*
    
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
main (int argc, char *argv[])
{
	if ( argc == 3 )
 	{
 		if (reader.read (argv[1], *scene) < 0)
 		{
    		std::cout << "Error loading scene cloud." << std::endl;
    		exit (0);
  		}
  		std::string fileName(argv[1]);
  		importObjectsInformation(argv[2]);
 		displayObjects();
 		std::string folderName = createDir(fileName);
 		extractPCD(_objectList, folderName);
 	}

 	else
    {
 		std::cout<<"Usage: " << argv[0] << " <filename>.pcd <filename>.xml" << std::endl;
        exit(0);
    }

}
