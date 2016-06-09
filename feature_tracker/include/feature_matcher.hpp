#ifndef VIO_FEATURE_MATCHER_
#define VIO_FEATURE_MATCHER_

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "../../mapdata/include/image_frame.hpp"

namespace vio {

enum FeatureMatchMethod {
  BRUTE_FORCE
};

class FeatureMatcher {
 public:
  FeatureMatcher()
      : max_match_per_desc_(2),
        nn_match_ratio_(0.9f) {}

  virtual bool Match(const ImageFrame &frame0, const ImageFrame &frame1,
                     std::vector<cv::DMatch> &matches) = 0;
 protected:
  // TODO: Right now, it's O(n^2) search time.
  bool SymmetryTestFilter(const std::vector<cv::DMatch> &matches1,
                          const std::vector<cv::DMatch> &matches2,
                          std::vector<cv::DMatch> &final_matches);
  bool RatioTestFilter(std::vector<std::vector<cv::DMatch> > best_k,
                       std::vector<cv::DMatch> &matches);

  double nn_match_ratio_;
  int max_match_per_desc_;
};

}  // vio

#endif
