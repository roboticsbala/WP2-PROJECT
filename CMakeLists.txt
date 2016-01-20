cmake_minimum_required(VERSION 2.8.3)
project(WP2-PROJECT)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)
find_package(PCL 1.5 REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

add_executable (correspondence_grouping_SHOT  src/correspondence_grouping_SHOT.cpp)
target_link_libraries (correspondence_grouping_SHOT ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable (correspondence_grouping_SHOT_Obj-Obj src/correspondence_grouping_SHOT_Obj-Obj.cpp)
target_link_libraries (correspondence_grouping_SHOT_Obj-Obj ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable (correspondence_grouping_FPFH  src/correspondence_grouping_FPFH.cpp)
#target_link_libraries (correspondence_grouping_FPFH ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable (cluster_extraction  src/cluster_extraction.cpp)
#target_link_libraries (cluster_extraction ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable (cluster_extraction_v2  src/cluster_extraction_v2.cpp)
#target_link_libraries (cluster_extraction_v2 ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable (directoryScan  src/directoryScan.cpp)
#target_link_libraries (directoryScan ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable (correspondence_grouping_SHOT_Iterative_Obj-Scene  src/correspondence_grouping_SHOT_Iterative_Obj-Scene.cpp)
#target_link_libraries (correspondence_grouping_SHOT_Iterative_Obj-Scene ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable (pcd_size  src/pcd_size.cpp)
target_link_libraries (pcd_size ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable (objectExtractor  src/objectExtractor.cpp)
target_link_libraries (objectExtractor ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable (correspondence_grouping_SHOT_Iterative_Obj-Scene_v2 src/correspondence_grouping_SHOT_Iterative_Obj-Scene_v2.cpp)
target_link_libraries (correspondence_grouping_SHOT_Iterative_Obj-Scene_v2 ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable (correspondence_grouping_CSHOT src/correspondence_grouping_CSHOT.cpp)
#target_link_libraries (correspondence_grouping_CSHOT ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable (3DSIFT_Keypoints src/3DSIFT_Keypoints.cpp)
#target_link_libraries (3DSIFT_Keypoints ${PCL_LIBRARIES})

add_executable (correspondence_grouping_SHOT_Iterative_Obj-Scene_v3 src/correspondence_grouping_SHOT_Iterative_Obj-Scene_v3.cpp)
target_link_libraries (correspondence_grouping_SHOT_Iterative_Obj-Scene_v3${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable (objectExtractorIterative src/objectExtractorIterative.cpp)
target_link_libraries (objectExtractorIterative ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable (correspondence_grouping_SHOT_Iterative_Obj-Obj  src/correspondence_grouping_SHOT_Iterative_Obj-Obj.cpp)
target_link_libraries (correspondence_grouping_SHOT_Iterative_Obj-Obj ${catkin_LIBRARIES} ${PCL_LIBRARIES})
