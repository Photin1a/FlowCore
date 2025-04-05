#ifndef STC_GEN_HPP
#define STC_GEN_HPP

#include <ompl/util/Console.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/DiscreteMotionValidator.h>
#include <ompl/geometric/SimpleSetup.h>

#include <memory>
#include <Eigen/Eigen>

#include <decomp_util/ellipsoid_decomp.h>

namespace stc_gen{
    class STCGen{
    public:
        // RRTConnect
        // validity_check_rate: normal is 1/grid_num
        static inline double PlanPath(const Eigen::Vector2d &start,
                            const Eigen::Vector2d &goal,
                            const Eigen::Vector2d &lb,
                            const Eigen::Vector2d &hb,
                            const double &validity_check_rate,
                            const double &timeout,
                            const std::function<bool(const ompl::base::State*)> &valid_checker_func,
                            std::vector<Eigen::Vector2d> &path
                            ){
            auto space = std::make_shared<ompl::base::RealVectorStateSpace>();
            space->addDimension(lb(0), hb(0));
            space->addDimension(lb(1), hb(1));

            auto setup = std::make_shared<ompl::geometric::SimpleSetup>(space);
            setup->setStateValidityChecker(valid_checker_func);
            space->setup();

            // printf("res %f\n",validity_check_rate);
            setup->getSpaceInformation()->setStateValidityCheckingResolution(validity_check_rate);
            // printf("max %f\n",space->getMaximumExtent());
            setup->setPlanner(std::make_shared<ompl::geometric::RRTConnect>(setup->getSpaceInformation()));

            printf("%f %f %f %f\n",lb(0),hb(0),lb(1),hb(1));

            ompl::base::ScopedState<> s(setup->getStateSpace()), g(setup->getStateSpace());
            s[0] = start[0];
            s[1] = start[1];
            g[0] = goal[0];
            g[1] = goal[1];
            printf("RRTConnect planning, start(%f %f), goal(%f %f)\n",start(0),start(1),goal(0),goal(1));
            setup->setStartAndGoalStates(s, g);

            double cost = INFINITY;
            setup->getPlanner()->clear();
            auto solved = setup->solve();   

            const std::size_t ns = setup->getProblemDefinition()->getSolutionCount();
            // OMPL_INFORM("Found %d solutions", (int)ns);
            if (setup->haveSolutionPath())
            {
                ompl::geometric::PathGeometric &path_t = setup->getSolutionPath();
                path_t.interpolate(20);
                for (size_t i = 0; i < path_t.getStateCount(); i++){
                    const auto state = path_t.getState(i)->as<ompl::base::RealVectorStateSpace::StateType>()->values;
                    path.emplace_back(state[0], state[1]);
                }
                printf("path searched, points: %d\n",path.size());
                cost = path_t.length();
            }
            return cost;
        }     
        
        // hpoly = [A,b]
        // line_segment = [p1,p2]
        static inline void ConvexHull(const std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d>> &line_segment,
                                const std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d>> &point_cloud,
                                Eigen::MatrixX3d &hpoly,
                                const double max_aaxis = 8.0,
                                const double max_baxis = 8.0){

            auto line = std::make_shared<LineSegment<2>>(line_segment[0], line_segment[1]);
            line->set_local_bbox(Eigen::Vector2d(max_aaxis,max_baxis));
            line->set_obs(point_cloud);
            line->dilate(0);
            
            auto lc2d = LinearConstraint2D((line_segment[0]+line_segment[1])/2,line->get_polyhedron().hyperplanes());
            hpoly.resize(lc2d.A_.rows(),3);
            hpoly << lc2d.A_,lc2d.b_;
        }

        // hpoly = [A,b]
        // line_segment = [p1,p2]
        // poly_vis
        static inline void ConvexHull(const std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d>> &line_segment,
                                const std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d>> &point_cloud,
                                Eigen::MatrixX3d &hpoly,Polyhedron<2> &poly_vis,
                                const double max_aaxis = 8.0,
                                const double max_baxis = 8.0){

            auto line = std::make_shared<LineSegment<2>>(line_segment[0], line_segment[1]);
            line->set_local_bbox(Eigen::Vector2d(max_aaxis,max_baxis));
            line->set_obs(point_cloud);
            line->dilate(0);
            
            poly_vis = line->get_polyhedron();

            auto lc2d = LinearConstraint2D((line_segment[0]+line_segment[1])/2,line->get_polyhedron().hyperplanes());
            hpoly.resize(lc2d.A_.rows(),3);
            hpoly << lc2d.A_,lc2d.b_;
        }

    };
}; // namespace stc_gen
#endif