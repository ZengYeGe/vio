#ifndef VIO_FEATURE_TRACKER_
#define VIO_FEATURE_TRACKER_

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "../../mapdata/include/image_frame.hpp"

#include "feature_matcher.hpp"

namespace vio {

enum FeatureTrackerMethod {
  OCV_BASIC_DETECTOR = 0,
  OCV_BASIC_DETECTOR_EXTRACTOR,
  SEARCH_BY_PROJECTION
};

struct FeatureTrackerOptions {
  FeatureTrackerOptions ()
      : method(OCV_BASIC_DETECTOR_EXTRACTOR),
        detector_type("ORB"),
        max_num_feature(10000),
        descriptor_type("DAISY") {}

  FeatureTrackerMethod method;

  // Detector
  std::string detector_type;
  int max_num_feature;

  // Descriptor
  std::string descriptor_type;
  
};


class FeatureTracker {
 public:
  FeatureTracker() {}
  ~FeatureTracker() {}

  static FeatureTracker *CreateFeatureTracker(
    FeatureTrackerOptions option);

  static FeatureTracker *CreateFeatureTrackerOCV(FeatureTrackerOptions option);

  virtual bool TrackFirstFrame(ImageFrame &output_frame) = 0;
  // TODO: Might need to use customized Match class.
  virtual bool TrackFrame(const ImageFrame &prev_frame,
                          ImageFrame &output_frame,
                          std::vector<cv::DMatch> &matches) = 0;

 protected:
  FeatureMatcher *matcher_;
};
}  // vio

#endif
