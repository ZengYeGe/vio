#include "feature_tracker_ocv.hpp"

#include "feature_matcher_ocv.hpp"

namespace vio {

FeatureTracker *FeatureTracker::CreateFeatureTracker(
    cv::Ptr<cv::FeatureDetector> detector) {
  FeatureTracker *tracker = new FeatureTrackerOCV(detector);
  return tracker;
}

FeatureTracker *FeatureTracker::CreateFeatureTracker(
    cv::Ptr<cv::FeatureDetector> detector,
    cv::Ptr<cv::DescriptorExtractor> extractor) {
  FeatureTracker *tracker = new FeatureTrackerOCV(detector, extractor);
  return tracker;
}

FeatureTrackerOCV::FeatureTrackerOCV(cv::Ptr<cv::FeatureDetector> detector) {
  detector_ = detector;
  detector_type_ = DETECTORONLY;
  InitTracker();
}

FeatureTrackerOCV::FeatureTrackerOCV(
    cv::Ptr<cv::FeatureDetector> detector,
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
bool FeatureTrackerOCV::TrackFrame(const Frame &prev_frame, Frame &new_frame,
                                   std::vector<cv::DMatch> &matches) {
  ComputeFeatures(new_frame);
  matcher_->Match(prev_frame.keypoints(), new_frame.keypoints(),
                  prev_frame.descriptors(), new_frame.descriptors(), matches);
  return true;
}

void FeatureTrackerOCV::InitTracker() { matcher_ = new FeatureMatcherOCV(); }

void FeatureTrackerOCV::ComputeFeatures(Frame &frame) {
  if (detector_type_ == DETECTORONLY) {
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    detector_->detectAndCompute(frame.GetImage(), cv::noArray(), kp, desc);

    frame.set_keypoints(kp);
    frame.set_descriptors(desc);
  } else if (detector_type_ == DETECTORDESCRIPTOR) {
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    detector_->detect(frame.GetImage(), kp);
    extractor_->compute(frame.GetImage(), kp, desc);

    frame.set_keypoints(kp);
    frame.set_descriptors(desc);
  }
}

}  // vio
