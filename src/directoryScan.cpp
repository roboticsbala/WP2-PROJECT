#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <iostream>
#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

int main( int argc, char* argv[] )
{ 
  fs::path full_path( fs::initial_path<fs::path>());

  if ( argc > 1 )
  {
    full_path = fs::system_complete( fs::path( argv[1]));
  }

  else
  {
    std::cout << "\nUsage: directoryScan [path]" << std::endl;
    return 0;
  }

  if (!fs::exists(full_path))
  {
    std::cout << "\nNot found: " << full_path.string() << std::endl;
    return 0;
  }

  if (fs::is_directory(full_path))
  { 
    //std::vector<fs::path> ret;
    unsigned long file_count = 0;
    std::cout<<"Is a valid directory"<<std::endl;
    const std::string ext = ".pcd";
    fs::recursive_directory_iterator it(full_path);
    fs::recursive_directory_iterator endit;

    while(it != endit)
    {  
      if(fs::is_regular_file(*it) && it->path().extension() == ext) 
      { 
       std::cout<<it->path().filename()<<std::endl;
       ++file_count;
        //ret.push_back(it->path().filename());
      }
      ++it;
    }
  }

  return 1;
}