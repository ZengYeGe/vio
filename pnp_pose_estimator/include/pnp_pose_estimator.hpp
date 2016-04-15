#include <vector>

#include <opencv2/opencv.hpp>

namespace vio {

class PnPPoseEstimator {
 public:
  PnPPoseEstimator() {}

  virtual bool EstimatePose(const std::vector<cv::Vec2d> &image_points,
                            const std::vector<cv::Point3d> &points3d,
                            cv::Mat &R_est, cv::Mat &t_est) = 0;

 protected:
};

} // vio
