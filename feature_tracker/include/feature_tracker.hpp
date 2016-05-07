#ifndef VIO_FEATURE_TRACKER_
#define VIO_FEATURE_TRACKER_

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "feature_matcher.hpp"

namespace vio {

enum FeatureTrackerMethod {
  OCV_BASIC_DETECTOR = 0,
  OCV_BASIC_DETECTOR_EXTRACTOR,
  SEARCH_BY_PROJECTION
};

class FeatureTrackerOptions {
 public:
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
  
  void read(const cv::FileNode& node) {
    method = static_cast<FeatureTrackerMethod>((int)node["Method"]);
    detector_type = (std::string)node["DetectorType"];
    max_num_feature = (int)node["max_num_feature"];

    descriptor_type = (std::string)node["DescriptorType"];
  }
  
};

// Following must be defined for the serialization in FileStorage to work
static void read(const cv::FileNode& node, FeatureTrackerOptions& x,
                 const FeatureTrackerOptions& default_value = FeatureTrackerOptions()){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}


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
};
}  // vio

#endif
