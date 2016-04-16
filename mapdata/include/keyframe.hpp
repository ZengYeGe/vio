#include <opencv2/opencv.hpp>

namespace vio {

class KeyFrame {
 public:
  // TODO: Really need to copy the image?
  explicit KeyFrame(const cv::Mat &image) { image.copyTo(image_); }

  const cv::Mat &GetImage() const { return image_; }

  void SetFeatures(FeatureSet &features) {
    features_.keypoints = std::move(features.keypoints);
    features.descriptors.copyTo(features_.descriptors);
  }

  void set_keypoints(std::vector<cv::KeyPoint> &keypoints) {
    keypoints_ = std::move(keypoints);
  }
  void set_descriptors(cv::Mat &descriptors) {
    descriptors_ = descriptors;
  }


 private:
  cv::Mat image_;

  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;

};

}  // namespace vio
