#include "feature_matcher_grid_search.hpp"

namespace vio {

bool FeatureMatcherGridSearch::Match(const ImageFrame &frame0, const ImageFrame &frame1,
                     std::vector<cv::DMatch> &matches) {
  const std::vector<cv::KeyPoint> &kp0 = frame0.keypoints();
  const std::vector<cv::KeyPoint> &kp1 = frame1.keypoints();
  const cv::Mat &desc0 = frame0.descriptors();
  const cv::Mat &desc1 = frame1.descriptors();

  return true;
}

} // vio


