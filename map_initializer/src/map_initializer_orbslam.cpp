#include "map_initializer_orbslam.hpp"

#include <iostream>

namespace vio {

MapInitializer *MapInitializer::CreateMapInitializerORBSLAM(MapInitializerOptions option) {
  MapInitializer *initializer = new MapInitializerORBSLAM(option);
  return initializer;
}

bool MapInitializerORBSLAM::Initialize(
    const std::vector<std::vector<cv::Vec2d> > &feature_vectors,
    const cv::Mat &K, std::vector<cv::Point3f> &points3d,
    std::vector<bool> &points3d_mask, std::vector<cv::Mat> &Rs,
    std::vector<cv::Mat> &ts) {

  if (feature_vectors.size() != 2) {
    std::cerr << "Error: F_or_H initializer only support two views.\n";
    return false;
  }

  return InitializeTwoFrames(feature_vectors[0], feature_vectors[1], K,
                             points3d, points3d_mask, Rs, ts);
}

} // vio
