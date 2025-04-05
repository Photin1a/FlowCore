cmake_minimum_required(VERSION 3.10)
project(minco_curve)

set(CMAKE_EXPORT_COMPILE_COMMANDS True)

set(CMAKE_BUILD_TYPE Release)

# list(APPEND CMAKE_PREFIX_PATH "/usr/local/share/ompl/cmake")
# Eigen
find_package(Eigen3 REQUIRED)
find_package(ompl 1.6.0 REQUIRED)

# glog
# include(cmake/glog.cmake)
# include_directories(${GLOG_INCLUDE_DIRS})

# catkin
find_package(catkin REQUIRED COMPONENTS
    roscpp
    nav_msgs
    tf
    decomp_ros_utils
    decomp_ros_msgs
    decomp_util
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

catkin_package(
     INCLUDE_DIRS include
     LIBRARIES PATH_SEARCHER_LIB
     CATKIN_DEPENDS nav_msgs roscpp tf   decomp_ros_utils decomp_ros_msgs decomp_util
    #  DEPENDS ${GLOG_LIBRARIES}
)

add_executable(test_node src/test_node.cpp)
target_link_libraries(test_node
    ${catkin_LIBRARIES}
    ompl::ompl
)
