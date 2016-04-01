#include "feature_tracker.hpp"

#include <opencv2/features2d.hpp>

namespace vio {

class FeatureTrackerOCV : public FeatureTracker {
 public:
  enum DetectorType {
    UNKNOWN = 0,
    DETECTORONLY,
    DETECTORDESCRIPTOR
  };

  FeatureTrackerOCV(cv::Ptr<cv::FeatureDetector> detector);
  FeatureTrackerOCV(cv::Ptr<cv::FeatureDetector> detector,
                    cv::Ptr<cv::DescriptorExtractor> extractor);

  FeatureTrackerOCV() = delete;

  virtual bool TrackFirstFrame(Frame &output_frame) override;
  virtual bool TrackFrame(const Frame &prev_frame,
                     Frame &output_frame,
                     std::vector<cv::DMatch> &matches) override;
 private:
  void InitTracker();
  void ComputeFeatures(Frame &frame);

  DetectorType detector_type_;
  cv::Ptr<cv::Feature2D> detector_;
  cv::Ptr<cv::DescriptorExtractor> extractor_;

};  
} // vio
