#include "landmark3d_initializer.hpp"

#include <iostream>

using namespace std;

Landmark3dInitializer::Landmark3dInitializer(bool is_proj)
    : is_projective_(is_proj),
      verbose_(true) {}

bool Landmark3dInitializer::initialize3DPointsFromViews(
    int num_frame,
    const vector<vector<cv::Vec2d> > &feature_vectors,
    const cv::Matx33d &initial_camera_matrix,
    vector<cv::Mat> &points3d_ests,
    vector<cv::Mat> &R_ests, vector<cv::Mat> &t_ests,
    cv::Matx33d &refined_camera_matrix) {
  if (num_frame != feature_vectors.size() ||
      num_frame < 2) return false;

  vector<cv::Mat> all_2d_points;
  const int num_features = feature_vectors[0].size();
  for (int i = 0; i < num_frame; ++i) {
    if (num_features != feature_vectors[i].size()) return false;

    cv::Mat_<double> frame(2, num_features);
    for (int j = 0; j < num_features; ++j) {
      frame(0, j) = feature_vectors[i][j][0];
      frame(1, j) = feature_vectors[i][j][1];
    }
    all_2d_points.push_back(cv::Mat(frame));
  }

  refined_camera_matrix = cv::Mat(initial_camera_matrix).clone();
  cv::sfm::reconstruct(all_2d_points, R_ests, t_ests, refined_camera_matrix,
                       points3d_ests, is_projective_);

  if (verbose_) {
    cout << "\n----------------------------\n" << endl;
    cout << "2D feature number: " << feature_vectors[0].size() << endl;
    cout << "Initialized 3D points: " << points3d_ests.size() << endl;
    cout << "Estimated cameras: " << R_ests.size() << endl;
    cout << "Original intrinsics: " << endl << initial_camera_matrix << endl;
    cout << "Refined intrinsics: " << endl << refined_camera_matrix << endl << endl;
    cout << "Cameras are: " << endl;
    for (int i = 0; i < R_ests.size(); ++i) {
      cout << "R: " << endl << R_ests[i] << endl;
      cout << "t: " << endl << t_ests[i] << endl;
    }
    cout << "\n----------------------------\n" << endl;
  }

  return true;
}

bool Landmark3dInitializer::initialize3DPointsFromViews(
    const vector<vector<cv::Vec2d> > &feature_vectors,
    const cv::Matx33d &inital_camera_matrix,
    vector<cv::Mat> &points3d_ests,
    vector<cv::Mat> &pose_ests,
    cv::Mat &refined_camera_matrix) {

  return false;
}
  

