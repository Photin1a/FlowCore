cmake_minimum_required(VERSION 3.0.0)
project(decomp_ros_utils)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_AUTOMOC ON)

find_package(catkin REQUIRED COMPONENTS rviz roscpp decomp_util decomp_ros_msgs)
find_package(Eigen3 REQUIRED)
include_directories(
  include src
  ${EIGEN3_INCLUDE_DIRS} 
  ${catkin_INCLUDE_DIRS}
  )

## We'll use the version that rviz used so they are compatible.
message("Using Qt5 based on the rviz_QT_VERSION: ${rviz_QT_VERSION}")
find_package(Qt5 ${rviz_QT_VERSION} REQUIRED COMPONENTS Core Widgets)
list(APPEND QT_LIBRARIES Qt5::Widgets Qt5::Core)

catkin_package(
  INCLUDE_DIRS include src ${EIGEN3_INCLUDE_DIRS} 
  LIBRARIES decomp_rviz_plugins
  CATKIN_DEPENDS decomp_util decomp_ros_msgs roscpp tf rviz 
  DEPENDS ${QT_LIBRARIES}
)

add_library(decomp_rviz_plugins   
  src/bound_visual.cpp
  src/mesh_visual.cpp
  src/vector_visual.cpp
  src/ellipsoid_array_visual.cpp
  src/ellipsoid_array_display.cpp
  src/polyhedron_array_display.cpp
  src/bound_visual.h
  src/mesh_visual.h
  src/vector_visual.h
  src/ellipsoid_array_visual.h
  src/ellipsoid_array_display.h
  src/polyhedron_array_display.h
)
add_dependencies(decomp_rviz_plugins ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(decomp_rviz_plugins ${QT_LIBRARIES} ${catkin_LIBRARIES} ${rviz_DEFAULT_PLUGIN_LIBRARIES})

