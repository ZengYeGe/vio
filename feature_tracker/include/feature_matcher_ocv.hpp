#ifndef VIO_FEATURE_MATCHER_OCV_
#define VIO_FEATURE_MATCHER_OCV_

#include "feature_matcher.hpp"

#include <vector>

namespace vio {

class FeatureMatcherOCV : public FeatureMatcher {
 public:
  FeatureMatcherOCV() : max_match_per_desc_(2), nn_match_ratio_(0.9f) {
    // TODO: Decide matcher based on descriptors
    // Hamming-distance works only for binary feature-types like ORB, FREAK
    // matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher_ = cv::DescriptorMatcher::create("BruteForce");
    // matcher_ = cv::DescriptorMatcher::create("FlannBased");
  }
  ~FeatureMatcherOCV(){};

  virtual bool Match(const ImageFrame &frame0, const ImageFrame &frame1,
                     std::vector<cv::DMatch> &matches);
 private:
  // TODO: Right now, it's O(n^2) search time.
  bool SymmetryTestFilter(const std::vector<cv::DMatch> &matches1,
                          const std::vector<cv::DMatch> &matches2,
                          std::vector<cv::DMatch> &final_matches);
  bool RemoveOutlierMatch(const std::vector<cv::KeyPoint> &pre_kp,
                          const std::vector<cv::KeyPoint> &cur_kp,
                          std::vector<cv::DMatch> &matches);

  cv::Ptr<cv::DescriptorMatcher> matcher_;
  int max_match_per_desc_;
};

}  // vio

#endif
