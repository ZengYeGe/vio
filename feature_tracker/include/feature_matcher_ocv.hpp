#ifndef VIO_FEATURE_MATCHER_OCV_
#define VIO_FEATURE_MATCHER_OCV_

#include "feature_matcher.hpp"

#include <vector>

namespace vio {

class FeatureMatcherOCV : public FeatureMatcher {
 public:
  FeatureMatcherOCV() {
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
  bool RemoveOutlierMatch(const std::vector<cv::KeyPoint> &pre_kp,
                          const std::vector<cv::KeyPoint> &cur_kp,
                          std::vector<cv::DMatch> &matches);

  cv::Ptr<cv::DescriptorMatcher> matcher_;
};

}  // vio

#endif
