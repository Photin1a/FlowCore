#ifndef TRAJ_OPTMAIZER_HPP
#define TRAJ_OPTMAIZER_HPP

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <nav_msgs/Path.h>
#include <tf2/utils.h>

#include "utils/lbfgs.hpp"
#include "utils/minco.hpp"
#include "stc_gen.hpp"

#include <decomp_ros_msgs/PolyhedronArray.h>
#include <decomp_ros_utils/data_ros_utils.h>

#include <Eigen/Eigen>
#include <algorithm>

#include "traj_optmizer.hpp"

class PathPlannerSim{
public:
    template<typename T>
    using VecE = std::vector<T,Eigen::aligned_allocator<T>>;

    int pieceNum_;
    int optDim_;
    int iter;
private:
    ros::NodeHandle &public_nh;
    nav_msgs::OccupancyGridConstPtr costmap_;

    bool has_initpose_,has_endpose_,has_map_;
    geometry_msgs::PoseStamped initpose_,endpose_;
    ros::Subscriber initpose_sub_, endpose_sub_, map_sub_;
    ros::Publisher path_pub_,traj_pub_;
    ros::Publisher stc_pub_;

    VecE<Eigen::MatrixX3d> hpolys;
    minco::MINCO_S3NU<2> opt_;
    Eigen::VectorXd ts_; 

    double max_radius_;

    void InitPoseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &initpose);
    void EndPoseCallback(const geometry_msgs::PoseStampedConstPtr &endpose);
    void MapCallback(const nav_msgs::OccupancyGridConstPtr &costmap);

public:
    // construct func
    // max_radius: max turning radius to car, default = 10.0
    PathPlannerSim(ros::NodeHandle &nh, double max_radius = 10.0):public_nh(nh),max_radius_(max_radius){
        initpose_sub_=public_nh.subscribe("/initialpose",1,&PathPlannerSim::InitPoseCallback,this);
        endpose_sub_=public_nh.subscribe("/move_base_simple/goal",1,&PathPlannerSim::EndPoseCallback,this);
        map_sub_ = public_nh.subscribe("/map",1,&PathPlannerSim::MapCallback,this);
        path_pub_ = public_nh.advertise<nav_msgs::Path>("/search_path",1);
        traj_pub_ = public_nh.advertise<nav_msgs::Path>("/opt_traj",1);
        stc_pub_ = public_nh.advertise<decomp_ros_msgs::PolyhedronArray>("/stc",1);
    }

private:
    // decompose velocity into x/y components
    // theta: velocity direction（maybe yaw）
    void DecompVel(const double theta, const double vel, double &vx, double &vy){
        const double epsi = 1e-8;
        double vt = std::abs(vel) < epsi?epsi:std::abs(vel);
        vx = vt*std::cos(theta);
        vy = vt*std::sin(theta);
    }

    // planning
    void Planning(const geometry_msgs::PoseStamped &initpose,const geometry_msgs::PoseStamped &endpose){
        double lbx = costmap_->info.origin.position.x;
        double ubx = costmap_->info.width*costmap_->info.resolution + lbx;
        double lby = costmap_->info.origin.position.y;
        double uby = costmap_->info.height*costmap_->info.resolution + lby;
        ROS_INFO("SimpleMoveBase::Start plan boundx(%f,%f),boundy(%f,%f)",lbx,ubx,lby,uby);
        std::vector<Eigen::Vector2d> path;
        auto cost = stc_gen::STCGen::PlanPath(Eigen::Vector2d(initpose.pose.position.x,initpose.pose.position.y),
            Eigen::Vector2d(endpose.pose.position.x,endpose.pose.position.y),
            Eigen::Vector2d(lbx,lby),Eigen::Vector2d(ubx,uby),1.0/(costmap_->info.width*costmap_->info.height),0.01,
            std::bind(&PathPlannerSim::Valid,this,std::placeholders::_1),path);

        pieceNum_ = path.size()-1;
        optDim_ = 2*(pieceNum_-1);

        Visua(path);
        VecE<Polyhedron<2>> ploys_vis;
        STCGen(costmap_,path,hpolys,ploys_vis);
        Visua(ploys_vis);

        double vsx,vsy,vex,vey;
        DecompVel(tf2::getYaw(initpose.pose.orientation),10,vsx,vsy);
        DecompVel(tf2::getYaw(endpose.pose.orientation),10,vex,vey);
        printf("vs(%f %f), ve(%f %f)\n",vsx,vsy,vex,vey);

        Eigen::Matrix<double, 3, 2> headState;
        headState.row(0) = path.front().transpose();
        headState.row(1) = Eigen::Vector2d(vsx,vsy).transpose();
        headState.row(2) = Eigen::Vector2d(0,0).transpose();
        Eigen::Matrix<double, 3, 2> tailState;
        tailState.row(0) = path.back().transpose();
        tailState.row(1) = Eigen::Vector2d(vex,vey).transpose();
        tailState.row(2) = Eigen::Vector2d(0,0).transpose();
        opt_.setConditions(headState,tailState,pieceNum_);

        Eigen::Matrix<double,2, -1> inPos(2,pieceNum_-1);
        inPos.setZero();
        for(int i = 1;i<pieceNum_;i++){
            inPos.col(i-1) = path[i];
        }
        Eigen::VectorXd xi(2*(pieceNum_-1));
        xi.topRows(pieceNum_-1) = inPos.row(0).transpose();
        xi.bottomRows(pieceNum_-1) = inPos.row(1).transpose();

        auto time1 = ros::Time::now();
        auto optcost = Lbfgs(xi);
        auto dt = ros::Time::now() - time1;
        printf("opt time: %f ms\n",dt.toSec()*1000);
        printf("opt cost: %.10f iter: %d\n",optcost,iter);

        inPos.row(0) = xi.topRows(pieceNum_-1).transpose();
        inPos.row(1) = xi.bottomRows(pieceNum_-1).transpose();

        opt_.setConditions(headState,tailState,pieceNum_);
        ts_ = Eigen::VectorXd::Constant(pieceNum_,1); // normalized time
        opt_.setParameters(inPos.transpose(),ts_);

        Trajectory<2,5> traj;
        opt_.getTrajectory(traj);

        // visualize sample points
        // "1" represents normalized time
        std::vector<Eigen::Vector2d> trajs;
        for(double t = 0.0;t<=traj.getPieceNum()*1;t+=0.05){
            trajs.emplace_back(traj.getPos(t));
        }
        VisuaTraj(trajs);
    }

    // lbfgs setup
    double Lbfgs(Eigen::VectorXd &xi){
        lbfgs::lbfgs_parameter_t opt_params;
        opt_params.max_linesearch = 128;
        opt_params.min_step = 1e-32;
        opt_params.mem_size = 128;
        opt_params.past = 3;
        opt_params.delta = 1.0e-6;
        opt_params.g_epsilon = 1.0e-5;
        opt_params.max_iterations = 10000;

        double cost = 0.0;
        int reslut_code = lbfgs::lbfgs_optimize(xi, cost,
                                &PathPlannerSim::costFunctional,
                                nullptr,
                                nullptr,
                                this,
                                opt_params);
        printf("lbfgs-ret: %s\n",lbfgs::lbfgs_strerror(reslut_code));
        return cost;
    }

    // relaxation/exterior penalty func
    static void positiveSmoothedL1(const double &x, double &f, double &df){
            const double pe = 1.0e-4;
            const double half = 0.5 * pe;
            const double f3c = 1.0 / (pe * pe);
            const double f4c = -0.5 * f3c / pe;
            const double d2c = 3.0 * f3c;
            const double d3c = 4.0 * f4c;

            if (x < pe)
            {
                f = (f4c * x + f3c) * x * x * x;
                df = (d3c * x + d2c) * x * x;
            }
            else
            {
                f = x - half;
                df = 1.0;
            }
            return;
    }

    // costfunc, callback-func of lbfgs
    static inline double costFunctional(void *ptr, const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
        PathPlannerSim &obj = *(PathPlannerSim *)ptr;
        const int dimXi = obj.optDim_;
        const int N = obj.pieceNum_;
        const double weightT = 1.0;

        obj.iter++;

        Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4;

        auto &ts = obj.ts_;

        double t2,t3,t4,t5;
        Eigen::Vector2d sigma, dsigma, ddsigma, dddsigma, ddddsigma;

        double gCur = 0.0;
        Eigen::Matrix<double, -1, 2> gradByC(6*N,2);
        gradByC.setZero();

        Eigen::Map<const Eigen::Matrix<double,2, -1, Eigen::RowMajor>> inPos(x.data(),2,dimXi/2);
        Eigen::Map<Eigen::Matrix<double, 2, -1, Eigen::RowMajor>> gradByP(grad.data(),2,dimXi/2);
        gradByP.setZero();

        Eigen::Matrix2d B;
        B << 0,-1,
             1, 0;

        obj.ts_ = Eigen::VectorXd::Constant(obj.pieceNum_,1); 
        obj.opt_.setParameters(inPos.transpose(),obj.ts_);
        
        for(int i = 0;i< N;i++){
            int Ki = 30; // 一段ki个点，包括端点
            double step = ts[i]/(Ki-1);

            const Eigen::Matrix<double, 6, 2> &c = obj.opt_.getCoeffs().middleRows(6*i,6);

            for(auto t1 = 0.0;t1 <= ts[i];t1+=step){
                if(i == 0 || i == N-1){
                    t1+=step;
                }
                t2 = t1*t1;
                t3 = t2*t1;
                t4 = t2*t2;
                t5 = t4*t1;

                beta0 << 1.0,t1,t2,t3,t4,t5;
                beta1 << 0.0, 1.0, 2.0 * t1, 3.0 * t2, 4.0 * t3, 5.0 * t4;
                beta2 << 0.0, 0.0, 2.0, 6.0 * t1, 12.0 * t2, 20.0 * t3;
                beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * t1, 60.0 * t2;
                beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120 * t1;

                sigma = c.transpose() * beta0;
                dsigma = c.transpose() * beta1;
                ddsigma = c.transpose() * beta2;
                dddsigma = c.transpose() * beta3;
                ddddsigma = c.transpose() * beta4;

                double z2 = ddsigma.transpose() * B * dsigma;
                auto vel = dsigma.norm();
                double epis = 1e-10;
                auto vel32 = std::pow(vel,3)+epis;
                auto vel62 = vel32*vel32;
                auto vel52 = std::pow(vel,5)+epis;

                double w_cur = 1;
                double w_cor = 1;

                Eigen::Matrix<double, 6, 2> gradByCT_t;
                gradByCT_t.setZero();

                //corridor 
                double violaCorPena = 0.0,violaCorPenaD = 0.0;
                auto resu = obj.hpolys[i].leftCols(2)*sigma-obj.hpolys[i].rightCols(1);
                for(int j = 0; j<resu.rows();j++){
                    if(resu(j)>0.0){
                        positiveSmoothedL1(resu(j), violaCorPena, violaCorPenaD);
                        gCur+= w_cor * violaCorPena;
                        gradByCT_t += w_cor*beta0*obj.hpolys[i].block(j,0,1,2)*violaCorPenaD;
                    }
                }

                // kappa


                double km =  1/obj.max_radius_;
                auto violaCurL = z2/vel32-km;
                auto violaCurR = -z2/vel32-km;
                double violaCurPenaL = 0,violaCurPenaDL = 0;
                double violaCurPenaR = 0,violaCurPenaDR = 0;

                if(violaCurL>0.0){
                    positiveSmoothedL1(violaCurL, violaCurPenaL, violaCurPenaDL);
                    gCur+=w_cur*violaCurPenaL;
                    gradByCT_t += w_cur*(beta1*(ddsigma.transpose()*B/vel32-3*z2*dsigma.transpose()/vel52)+
                    beta2*dsigma.transpose()*B.transpose()/vel32 )*violaCurPenaDL;

                }

                if(violaCurR>0.0){
                    positiveSmoothedL1(violaCurR, violaCurPenaR, violaCurPenaDR);
                    gCur+=w_cur*violaCurPenaR;
                    gradByCT_t += w_cur * -(beta1*(ddsigma.transpose()*B/vel32-3*z2*dsigma.transpose()/vel52)+
                    beta2*dsigma.transpose()*B.transpose()/vel32 )*violaCurPenaDR;
                }
                gradByC.middleRows(6*i,6) += gradByCT_t; 
            }
        }
                
        obj.opt_.propogateGrad(gradByC,gradByP);
        return gCur;
    }

    // Map frame to World frame
    void Map2World(const int mx, const int my, double &wx,double &wy){
        wx = costmap_->info.origin.position.x + (mx + 0.5) * costmap_->info.resolution;
        wy = costmap_->info.origin.position.y + (my + 0.5) * costmap_->info.resolution;
    }

    // get value of map-unit
    int GetMapItem(const int x,const int y){
        if(x<0 || x>=costmap_->info.width || y<0 || y>=costmap_->info.height)return -1;
        return costmap_->data[y*costmap_->info.width+x];
    }

    // World frame to Map frame
    void World2Map(const double wx,const double wy, int &mx, int &my){
        mx = static_cast<int>((wx - costmap_->info.origin.position.y) / costmap_->info.resolution);
        my = static_cast<int>((wy - costmap_->info.origin.position.y) / costmap_->info.resolution);
    }

    // validity check func(collision avoid)
    bool Valid(const ompl::base::State* state){
        const auto *pos = state->as<ompl::base::RealVectorStateSpace::StateType>();
        int x = 0;
        int y = 0;
        World2Map((*pos)[0],(*pos)[1],x,y);
        if(GetMapItem(x,y) == 100){
            return false;
        }
        return true;
    }

    void Visua(std::vector<Eigen::Vector2d> &path);

    // Visualize safety corridor
    void Visua(VecE<Polyhedron<2>> &ploys_vis){
        decomp_ros_msgs::PolyhedronArray poly_msg = polyhedron_array_to_ros(ploys_vis, 1);
        stc_pub_.publish(poly_msg);
    }

    template <int Dim>
    decomp_ros_msgs::PolyhedronArray polyhedron_array_to_ros(const vec_E<Polyhedron<Dim>>& vs, int delta = 1){
        decomp_ros_msgs::PolyhedronArray msg;
        msg.header.frame_id = "map";
        msg.header.stamp = ros::Time::now();
        int i = 1;
        for (const auto &v : vs){
            if(i == delta){
                msg.polyhedrons.push_back(DecompROS::polyhedron_to_ros(v)); 
                i = 0;   
            }
            i++;
        }
        return msg;
    }

    // generate  safe corridor  along ref-path(from rrtconnect)
    void STCGen(const nav_msgs::OccupancyGridConstPtr map,const std::vector<Eigen::Vector2d> &path,
        VecE<Eigen::MatrixX3d> &hpoly, VecE<Polyhedron<2>> &ploys_vis){
        hpoly.resize(path.size()-1);
        ploys_vis.resize(path.size()-1);
        for(int i = 0; i<path.size()-1;i++){
            auto center = (path[i]+path[i+1])/2;
            double dist = (path[i]-path[i+1]).norm();
            Eigen::Vector2d box(dist,dist);
            VecE<Eigen::Vector2d> vec_map;
            double side = std::pow(std::max(box.x(),box.y())*2,2);
            Map2Vec2D(costmap_,center.x(),center.y(),side,side,vec_map);
            ploys_vis.push_back(Polyhedron2D());
            stc_gen::STCGen::ConvexHull({path[i],path[i+1]},vec_map,hpoly[i],ploys_vis.back(),dist,dist);
        }
    }

    // from occ-map to point-cloud
    void Map2Vec2D(const nav_msgs::OccupancyGridConstPtr costmap,const double center_x, const double center_y, 
    const double roi_w,const double roi_h, VecE<Eigen::Vector2d> &vec_map){
        vec_map.clear();
        auto resolution = costmap->info.resolution;
        auto o_x = costmap->info.origin.position.x;
        auto o_y = costmap->info.origin.position.y;

        size_t i = 0;

        auto x_s = static_cast<int>((center_x - roi_w/2-o_x)/resolution);
        auto y_s = static_cast<int>((center_y - roi_h/2-o_y)/resolution);
        x_s = x_s < 0?0:x_s;
        y_s = y_s < 0?0:y_s;
        auto x_e = static_cast<int>((center_x + roi_w/2-o_x)/resolution);
        auto y_e = static_cast<int>((center_y + roi_h/2-o_y)/resolution);
        x_e = x_e < costmap->info.width?x_e:costmap->info.width-1;
        y_e = y_e < costmap->info.height?y_e:costmap->info.height-1;

        for (int i = x_s;i<= x_e;i++)
        for (int j = y_s;j<= y_e;j++) {
            if (costmap->data[j*costmap->info.width+i] > 0){
                vec_map.emplace_back((i+0.5) * resolution, (j+0.5) * resolution); 
            }
        }
    }

    // VisuaTraj
    void VisuaTraj(std::vector<Eigen::Vector2d> &path){
        nav_msgs::Path vis_path;
        vis_path.header.frame_id = "map";
        vis_path.header.stamp = ros::Time::now();
        vis_path.poses.resize(path.size());
        for(int i = 0;i<path.size();i++){
            vis_path.poses[i].header.frame_id = "map";
            vis_path.poses[i].header.stamp = ros::Time::now();
            vis_path.poses[i].pose.position.x = path[i].x();
            vis_path.poses[i].pose.position.y = path[i].y();
        }
        traj_pub_.publish(vis_path);
    }

};

// recv init-pose of rviz
void PathPlannerSim::InitPoseCallback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &initpose){
    ROS_INFO("SimpleMoveBase::InitPoseCallback");
    initpose_.header = initpose->header;
    initpose_.pose.position.x = initpose->pose.pose.position.x;
    initpose_.pose.position.y = initpose->pose.pose.position.y;
    initpose_.pose.orientation = initpose->pose.pose.orientation;
    has_initpose_=true;
}

void PathPlannerSim::Visua(std::vector<Eigen::Vector2d> &path){
    nav_msgs::Path vis_path;
    vis_path.header.frame_id = "map";
    vis_path.header.stamp = ros::Time::now();
    vis_path.poses.resize(path.size());
    for(int i = 0;i<path.size();i++){
        vis_path.poses[i].header.frame_id = "map";
        vis_path.poses[i].header.stamp = ros::Time::now();
        vis_path.poses[i].pose.position.x = path[i].x();
        vis_path.poses[i].pose.position.y = path[i].y();
    }
    path_pub_.publish(vis_path);
}

// recv goal-pose of rviz
void PathPlannerSim::EndPoseCallback(const geometry_msgs::PoseStampedConstPtr &endpose){
    ROS_INFO("SimpleMoveBase::EndPoseCallback");
    endpose_.header = endpose->header;
    endpose_.pose.position.x = endpose->pose.position.x;
    endpose_.pose.position.y = endpose->pose.position.y;
    endpose_.pose.orientation = endpose->pose.orientation;
    has_endpose_=true;
    if(has_initpose_&&has_endpose_&&has_map_){
        has_initpose_=false;
        has_endpose_=false;
        Planning(initpose_,endpose_);
    }
}

// recv occ-map of mapserver
void PathPlannerSim::MapCallback(const nav_msgs::OccupancyGridConstPtr &costmap){
    costmap_ = costmap;
    has_map_ = true;
    ROS_INFO("SimpleMoveBase::MapCallback");
}
#endif