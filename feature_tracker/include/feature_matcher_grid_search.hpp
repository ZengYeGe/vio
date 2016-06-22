#ifndef VIO_FEATURE_MATCHER_GRIDSEARCH_
#define VIO_FEATURE_MATCHER_GRIDSEARCH_

#include "feature_matcher.hpp"

#include <vector>

namespace vio {

class FeatureMatcherGridSearch : public FeatureMatcher {
 public:
  FeatureMatcherGridSearch(FeatureMatcherOptions option);

  bool Match(const ImageFrame &frame0, const ImageFrame &frame1,
                     std::vector<cv::DMatch> &matches) override;
 private:
  bool FindMatchNearFeatures(const ImageFrame &query_frame,
                             const ImageFrame &ref_frame,
                             std::vector<std::vector<cv::DMatch> > &matches);
  // TODO: Make it inline
  double ComputeDistance(const cv::Mat &mat0, const cv::Mat &mat1);
};


}  // vio

#endif
