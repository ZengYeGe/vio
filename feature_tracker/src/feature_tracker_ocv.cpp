#include "feature_tracker_ocv.hpp"

#include "feature_matcher_ocv.hpp"

namespace vio {

FeatureTracker *FeatureTracker::CreateFeatureTracker(cv::Ptr<cv::FeatureDetector> detector) {
  FeatureTracker *tracker = new FeatureTrackerOCV(detector);
  return tracker;
}

FeatureTracker *FeatureTracker::CreateFeatureTracker(cv::Ptr<cv::FeatureDetector> detector,
                                              cv::Ptr<cv::DescriptorExtractor> extractor) {
  FeatureTracker *tracker = new FeatureTrackerOCV(detector, extractor);
  return tracker;
}

FeatureTrackerOCV::FeatureTrackerOCV(cv::Ptr<cv::FeatureDetector> detector) {
  detector_ = detector;
  detector_type_ = DETECTORONLY;
  InitTracker();
} 
 
FeatureTrackerOCV::FeatureTrackerOCV(cv::Ptr<cv::FeatureDetector> detector,
                    cv::Ptr<cv::DescriptorExtractor> extractor) {
  detector_ = detector;
  extractor_ = extractor; 
  detector_type_ = DETECTORDESCRIPTOR;
  InitTracker();
}

bool FeatureTrackerOCV::TrackFirstFrame(Frame &output_frame) {
  ComputeFeatures(output_frame);
  return true;
}
bool FeatureTrackerOCV::TrackFrame(const Frame &prev_frame,
                Frame &new_frame,
               std::vector<cv::DMatch> &matches) {
  ComputeFeatures(new_frame);
  matcher_->Match(prev_frame.GetFeatures(), new_frame.GetFeatures(), matches);

  return true;
}

void FeatureTrackerOCV::InitTracker() {
  matcher_ = new FeatureMatcherOCV();
}

void FeatureTrackerOCV::ComputeFeatures(Frame &frame) {
  if (detector_type_ == DETECTORONLY) {
    FeatureSet features;
    detector_->detectAndCompute(frame.GetImage(), cv::noArray(),
                                features.keypoints, features.descriptors);
    frame.SetFeatures(features);
  } else if (detector_type_ == DETECTORDESCRIPTOR) {
    FeatureSet features;
    detector_->detect(frame.GetImage(), features.keypoints);
    extractor_->compute(frame.GetImage(), features.keypoints, features.descriptors);

    frame.SetFeatures(features);
  }
}
 
} // vio
