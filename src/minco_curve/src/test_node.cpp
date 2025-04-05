#include <ros/ros.h>
#include "traj_optmizer.hpp"

int main(int argc, char **argv){
    ros::init(argc, argv, "minco_curve");
    ros::NodeHandle nh;
    PathPlannerSim sim(nh,10.0);
    ros::spin();
    return 0;
}