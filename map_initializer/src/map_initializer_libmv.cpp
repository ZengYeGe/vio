#include "map_initializer_libmv.hpp"

#include <iostream>

namespace vio {

MapInitializer *MapInitializer::CreateMapInitializerLIBMV() {
  MapInitializer *initializer = new MapInitializerLIBMV();
  return initializer;
}

bool MapInitializerLIBMV::Initialize(const std::vector<std::vector<cv::Vec2d> > &feature_vectors,
                   const cv::Mat &K, std::vector<cv::Point3f> &points3d,
                   std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts) {
  if (feature_vectors.size() < 3) {
    std::cerr << "Error: libmv initializer not support views < 3.\n";
    return false;
  }
  
  std::vector<cv::Mat> all_2d_points;
  const int num_frame = feature_vectors.size();
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

  cv::Mat refined_camera_matrix = cv::Mat(K).clone();
  std::vector<cv::Mat> points3d_mat;
  cv::sfm::reconstruct(all_2d_points, Rs, ts, refined_camera_matrix, points3d_mat, true);

  // Convert mat to point3f
  points3d.clear();
  for (int i = 0; i < points3d_mat.size(); ++i) {
    points3d.push_back(cv::Point3f(points3d_mat.at(i)));
  }

  std::cout << "\n--------Initialization--------------------\n" << std::endl;
  std::cout << "2D feature number: " << feature_vectors[0].size() << std::endl;
  std::cout << "Initialized 3D points: " << points3d.size() << std::endl;
  std::cout << "Estimated cameras: " << Rs.size() << std::endl;
  std::cout << "Original intrinsics: " << std::endl
       << K << std::endl;
  std::cout << "Refined intrinsics: " << std::endl
       << refined_camera_matrix << std::endl
       << std::endl;
  std::cout << "Cameras are: " << std::endl;
  for (int i = 0; i < Rs.size(); ++i) {
    std::cout << "R: " << std::endl
         << Rs[i] << std::endl;
    std::cout << "t: " << std::endl
         << ts[i] << std::endl;
  }
  std::cout << "\n----------------------------\n" << std::endl;
  return true;
}
 
} // vio
