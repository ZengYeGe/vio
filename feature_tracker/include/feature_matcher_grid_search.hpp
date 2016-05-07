#ifndef VIO_FEATURE_MATCHER_GRIDSEARCH_
#define VIO_FEATURE_MATCHER_GRIDSEARCH_

#include "feature_matcher.hpp"

#include <vector>

namespace vio {

class FeatureMatcherGridSearch : public FeatureMatcher {
 public:
  FeatureMatcherGridSearch() {}

  virtual bool Match(const ImageFrame &frame0, const ImageFrame &frame1,
                     std::vector<cv::DMatch> &matches);
 private:
  // cv::Ptr<cv::DescriptorMatcher> matcher_;
};

}  // vio

#endif
