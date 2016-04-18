#ifndef VIO_FEATURE_TRACKER_
#define VIO_FEATURE_TRACKER_

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

#include "../../mapdata/include/image_frame.hpp"

#include "feature_matcher.hpp"

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

  virtual bool TrackFirstFrame(ImageFrame &output_frame) = 0;
  // TODO: Might need to use customized Match class.
  virtual bool TrackFrame(const ImageFrame &prev_frame, ImageFrame &output_frame,
                          std::vector<cv::DMatch> &matches) = 0;

 protected:
  FeatureMatcher *matcher_;
};
}  // vio

#endif
