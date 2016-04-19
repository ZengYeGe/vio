#ifndef VIO_IMAGE_FRAME_
#define VIO_IMAGE_FRAME_

#include <opencv2/opencv.hpp>

namespace vio {

class ImageFrame {
 public:
  ImageFrame(const cv::Mat &image) { image.copyTo(image_); }
  ImageFrame() = delete;

  const cv::Mat &GetImage() const { return image_; }

  void set_features(std::vector<cv::KeyPoint> &keypoints,
                    cv::Mat &descriptors) {
    set_keypoints(keypoints);
    set_descriptors(descriptors);
  }

  const std::vector<cv::KeyPoint> &keypoints() const { return keypoints_; }
  void set_keypoints(std::vector<cv::KeyPoint> &keypoints) {
    keypoints_ = std::move(keypoints);
  }
  const cv::Mat &descriptors() const { return descriptors_; }
  void set_descriptors(cv::Mat &descriptors) {
    descriptors.copyTo(descriptors_);
  }

 protected:
  cv::Mat image_;

  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;
};

}  // namespace vio

#endif
