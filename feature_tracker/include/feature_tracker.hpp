#ifndef VIO_FEATURE_TRACKER_
#define VIO_FEATURE_TRACKER_

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <vector>

#include "feature_matcher.hpp"
#include "frame.hpp"

namespace vio {

class FeatureTracker {
 public:
  FeatureTracker() {}
  ~FeatureTracker() {}

  static FeatureTracker *CreateFeatureTracker(
      cv::Ptr<cv::FeatureDetector> detector);
  static FeatureTracker *CreateFeatureTracker(
      cv::Ptr<cv::FeatureDetector> detector,
      cv::Ptr<cv::DescriptorExtractor> extractor);

  virtual bool TrackFirstFrame(Frame &output_frame) = 0;
  // TODO: Might need to use customized Match class.
  virtual bool TrackFrame(const Frame &prev_frame, Frame &output_frame,
                          std::vector<cv::DMatch> &matches) = 0;

 protected:
  FeatureMatcher *matcher_;
};
}  // vio

#endif
