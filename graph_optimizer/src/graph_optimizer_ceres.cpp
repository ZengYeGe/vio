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
  if (!ConstructProblem(K, Rs, ts, points))
    return false;

  ceres::Problem problem;
  const int num_obs = obs_feature.size();

  for (int i = 0; i < num_obs; ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.
    double observation[2];
    observation[0] = obs_feature[i][0];
    observation[1] = obs_feature[i][1];

    ceres::CostFunction* cost_function =
        SnavelyReprojectionError::Create(obs_feature[i][0],
                                         obs_feature[i][1]);
//    ceres::CostFunction* cost_function =
//        SnavelyReprojectionError::Create(observation[0],
//                                         observation[1]);
    problem.AddResidualBlock(cost_function,
                             NULL /* squared loss */,
                             cameras_[obs_camera_idx[i]].data(),
                             points_.data() + obs_point_idx[i] * 3);
  }   
  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 100;

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
  if (Rs.size() != ts.size() || !Rs.size() || !points.size())
    return false;
  const int num_cameras = Rs.size();
  const int num_points = points.size();
  cameras_.resize(num_cameras);
  points_.resize(num_points * 3);

  for (int i = 0; i < num_cameras; ++i) {
    cameras_[i].resize(9);

    double angle_axis[3];
    MatRotToAngleAxis(Rs[i], angle_axis);

    cameras_[i][0] = angle_axis[0];
    cameras_[i][1] = angle_axis[1];
    cameras_[i][2] = angle_axis[2];

    cameras_[i][3] = ts[i].at<double>(0);
    cameras_[i][4] = ts[i].at<double>(1);
    cameras_[i][5] = ts[i].at<double>(2);

    cameras_[i][6] = K.at<double>(0, 0);

    cameras_[i][7] = 0;
    cameras_[i][8] = 0;
  }

  for (int i = 0; i < num_points; ++i) {
    points_[i * 3 + 0] = points[i].x;
    points_[i * 3 + 1] = points[i].y;
    points_[i * 3 + 2] = points[i].z;
  }

  CheckPointsValid();
  
  return true;
}
 
bool GraphOptimizerCeres::AssignOptimizedResult(std::vector<cv::Mat> &Rs,
                              std::vector<cv::Mat> &ts,
                              std::vector<cv::Point3f> &points) {
  const int num_cameras = cameras_.size();
  const int num_points = points.size();

  for (int i = 0; i < num_cameras; ++i) {
    double angle_axis[3];
    angle_axis[0] = cameras_[i][0];
    angle_axis[1] = cameras_[i][1];
    angle_axis[2] = cameras_[i][2];

    AngleAxisToMatRot(angle_axis, Rs[i]);
    ts[i].at<double>(0) = cameras_[i][3];
    ts[i].at<double>(1) = cameras_[i][4];
    ts[i].at<double>(2) = cameras_[i][5];
  }

  // TODO: double to float
  for (int i = 0; i < num_points; ++i) {
    points[i].x = (float)points_[i * 3 + 0];
    points[i].y = (float)points_[i * 3 + 1];
    points[i].z = (float)points_[i * 3 + 2];
  }
  return true;
}

void GraphOptimizerCeres::CheckPointsValid() {
  for (int c_id = 0; c_id < cameras_.size(); ++c_id) {
    for (int p_id = 0; p_id < points_.size() / 3; ++p_id) {
      double p[3];
      ceres::AngleAxisRotatePoint(cameras_[c_id].data(),
                                  points_.data() + p_id * 3, p);
      if (p[2] == 0) {
        std::cout << "Error obs: camera " << c_id << " point " << p_id << std::endl;
      }
    } 
  }
}

void GraphOptimizerCeres::MatRotToAngleAxis(const cv::Mat &R_mat, double *angle_axis) {
  double R[9];
  // TODO: which order?
  R[0] = R_mat.at<double>(0, 0);
  R[3] = R_mat.at<double>(0, 1);
  R[6] = R_mat.at<double>(0, 2);
  R[1] = R_mat.at<double>(1, 0);
  R[4] = R_mat.at<double>(1, 1);
  R[7] = R_mat.at<double>(1, 2);
  R[2] = R_mat.at<double>(2, 0);
  R[5] = R_mat.at<double>(2, 1);
  R[8] = R_mat.at<double>(2, 2);
  ceres::RotationMatrixToAngleAxis(R, angle_axis);
}

void GraphOptimizerCeres::AngleAxisToMatRot(const double *angle_axis, cv::Mat &R_mat) {
  R_mat = cv::Mat(3, 3, CV_64F);
  double R[9];
  ceres::AngleAxisToRotationMatrix(angle_axis, R); 
  R_mat.at<double>(0, 0) = R[0];
  R_mat.at<double>(0, 1) = R[3];
  R_mat.at<double>(0, 2) = R[6];
  R_mat.at<double>(1, 0) = R[1];
  R_mat.at<double>(1, 1) = R[4];
  R_mat.at<double>(1, 2) = R[7];
  R_mat.at<double>(2, 0) = R[2];
  R_mat.at<double>(2, 1) = R[5];
  R_mat.at<double>(2, 2) = R[8];
}

} // vio
