#ifndef VIO_IMAGE_FRAME_
#define VIO_IMAGE_FRAME_

#include <opencv2/opencv.hpp>

namespace vio {

class ImageFrame {
 public:
  ImageFrame(const cv::Mat &image)
      : has_grid_keypoints_(false) { image.copyTo(image_); }
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
    CreateGridKeypointIndex();
  }
  const cv::Mat &descriptors() const { return descriptors_; }
  void set_descriptors(cv::Mat &descriptors) {
    descriptors.copyTo(descriptors_);
  }

 private:
  void CreateGridKeypointIndex();

  cv::Mat image_;

  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;

  bool has_grid_keypoints_;
  // Number of pixel of the width of a grid
  int grid_width_size_;
  // Number of pixel of the height of a grid
  int grid_height_size_;
  // image_width / grid_width_size_
  int grid_width_max_index_;
  // image_height / grid_height_size_
  int grid_height_max_index_;

  typedef std::vector<int> KeypointIndexArry;
  std::vector<std::vector<KeypointIndexArry> > grid_keypoints_index_;
};

}  // namespace vio

#endif
