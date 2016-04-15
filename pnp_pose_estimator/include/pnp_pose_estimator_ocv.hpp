#include "pnp_pose_estimator.hpp"

namespace vio {

class PnPPoseEstimatorOCV : public PnPPoseEstimator {
 public:
  PnPPoseEstimatorOCV() {}

  bool EstimatePose(const std::vector<cv::Vec2d> &image_points,
                            const std::vector<cv::Point3d> &points3d,
                            cv::Mat &R_est, cv::Mat &t_est) override;
};

} // vio
