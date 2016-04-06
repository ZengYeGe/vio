#include <opencv2/opencv.hpp>

#include "feature_set.hpp"

namespace vio {

class Frame {
 public:
  // TODO: Really need to copy the image?
  explicit Frame(const cv::Mat &image) { image.copyTo(image_); }

  const cv::Mat &GetImage() const { return image_; }

  void SetFeatures(FeatureSet &features) {
    features_.keypoints = std::move(features.keypoints);
    features.descriptors.copyTo(features_.descriptors);
  }

  const FeatureSet &GetFeatures() const { return features_; }

 private:
  cv::Mat image_;
  FeatureSet features_;
};

}  // namespace vio
