#include "feature_tracker_ocv.hpp"

#include "feature_matcher_ocv.hpp"

namespace vio {

FeatureTracker *FeatureTracker::CreateFeatureTrackerOCV(FeatureTrackerOptions option) {
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> descriptor;

  if (option.detector_type == "ORB") {
    detector = cv::ORB::create(option.max_num_feature);
  } else {
    return nullptr;
  }

  // If use descriptor
  if (option.method == OCV_BASIC_DETECTOR_EXTRACTOR) {
    if (option.descriptor_type == "DAISY") {
      descriptor = cv::xfeatures2d::DAISY::create();
    } else {
      return nullptr;
    }
  }

  switch (option.method) {
    case OCV_BASIC_DETECTOR:
      return new FeatureTrackerOCV(detector);
    case OCV_BASIC_DETECTOR_EXTRACTOR:
      return new FeatureTrackerOCV(detector, descriptor);
    default:
      return nullptr;
  }
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

bool FeatureTrackerOCV::TrackFirstFrame(ImageFrame &output_frame) {
  ComputeFeatures(output_frame);
  return true;
}
bool FeatureTrackerOCV::TrackFrame(const ImageFrame &prev_frame,
                                   ImageFrame &new_frame,
                                   std::vector<cv::DMatch> &matches) {
  ComputeFeatures(new_frame);
  if (!matcher_->Match(prev_frame.keypoints(), new_frame.keypoints(),
                  prev_frame.descriptors(), new_frame.descriptors(), matches))
    return false;
  return true;
}

void FeatureTrackerOCV::InitTracker() { matcher_ = new FeatureMatcherOCV(); }

void FeatureTrackerOCV::ComputeFeatures(ImageFrame &frame) {
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
