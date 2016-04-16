#ifndef VIO_FEATURE_MATCHER_
#define VIO_FEATURE_MATCHER_

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

namespace vio {

class FeatureMatcher {
 public:
  FeatureMatcher(){};
  ~FeatureMatcher(){};

  virtual bool Match(const std::vector<cv::KeyPoint> &kp0,
                     const std::vector<cv::KeyPoint> &kp1, const cv::Mat &desc0,
                     const cv::Mat &desc1,
                     std::vector<cv::DMatch> &matches) = 0;
};

}  // vio

#endif
