#ifndef CAMERA_MODEL_
#define CAMERA_MODEL_

#include <opencv2/opencv.hpp>

namespace vio {

enum CameraType {
  REGULAR,
  FISHEYE
};

class CameraModelParams {
 public:
  void read(const cv::FileNode& node) {
    type = static_cast<CameraType>((int)node["Type"]);
    node["K"] >> K;    
  }

  CameraType type;
  cv::Mat K;
  // TODO:
  // radial
  // tangential
};

// Following must be defined for the serialization in FileStorage to work
static void read(const cv::FileNode& node, CameraModelParams& x,
                 const CameraModelParams& default_value = CameraModelParams()){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

class CameraModel {
 public:
  CameraModel(CameraModelParams params) {
    type_ = params.type;
    params.K.copyTo(K_);
  }
  CameraModel() = delete;

  void UndistortImage(const cv::Mat &input, cv::Mat &undistort_img);

  const cv::Mat &K() const { return K_; }

 private:
  CameraType type_;
  cv::Mat K_;
};

}


#endif
