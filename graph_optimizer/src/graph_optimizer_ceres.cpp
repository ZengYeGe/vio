#include "graph_optimizer_ceres.hpp"

#include <iostream>

namespace vio {

GraphOptimizer *GraphOptimizer::CreateGraphOptimizerCeres() {
  GraphOptimizer *optimizer = new GraphOptimizerCeres();
  return optimizer;
}

bool GraphOptimizerCeres::Optimize(const cv::Mat &K,
                              std::vector<cv::Mat> &Rs,
                              std::vector<cv::Mat> &ts,
                              std::vector<cv::Point3f> &points,
                              const std::vector<int> &obs_camera_idx,
                              const std::vector<int> &obs_point_idx,
                              const std::vector<cv::Vec2d> &obs_feature) {
  ConstructProblem(K, Rs, ts, points);

  ceres::Problem problem;
  const int num_obs = obs_feature.size();

  for (int i = 0; i < num_obs; ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.
    double observation[2];
    ceres::CostFunction* cost_function =
        SnavelyReprojectionError::Create(observation[0],
                                         observation[1]);
    problem.AddResidualBlock(cost_function,
                             NULL /* squared loss */,
                             &cameras_[obs_camera_idx[i]],
                             &points_[obs_point_idx[i]]);
  }   
  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  AssignOptimizedResult(Rs, ts, points);

  return false;
}

bool GraphOptimizerCeres::ConstructProblem(const cv::Mat &K,
                              const std::vector<cv::Mat> &Rs,
                              const std::vector<cv::Mat> &ts,
                              const std::vector<cv::Point3f> &points) {
  const int num_cameras = Rs.size();
  const int num_points = points.size();
  cameras_.resize(num_cameras * 9);
  points_.resize(num_points * 3);

  for (int i = 0; i < num_cameras; ++i) {
    // cameras_[i * 9 + 0] = 
  }

  for (int i = 0; i < num_points; ++i) {
    points_[i * 3 + 0] = points[i].x;
    points_[i * 3 + 1] = points[i].y;
    points_[i * 3 + 2] = points[i].z;
  }
  
  return true;
}
 
bool GraphOptimizerCeres::AssignOptimizedResult(std::vector<cv::Mat> &Rs,
                              std::vector<cv::Mat> &ts,
                              std::vector<cv::Point3f> &points) {
  const int num_cameras = cameras_.size() / 9;
  const int num_points = points_.size() / 3;

  for (int i = 0; i < num_cameras; ++i) {
    // cameras_[i * 9 + 0] = 
  }

  for (int i = 0; i < num_points; ++i) {
    points[i].x = points_[i * 3 + 0];
    points[i].y = points_[i * 3 + 1];
    points[i].z = points_[i * 3 + 2];
  }
   return true;
}

} // vio
