#ifndef VIO_FEATURE_MATCHER_
#define VIO_FEATURE_MATCHER_

#include "feature_set.hpp"

namespace vio {

class FeatureMatcher {
 public:
  FeatureMatcher(){};
  ~FeatureMatcher(){};

  virtual bool Match(const FeatureSet &features0, const FeatureSet &features1,
                     std::vector<cv::DMatch> &matches) = 0;
};

}  // vio

#endif
