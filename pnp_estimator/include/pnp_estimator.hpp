#include <vector>

#include <opencv2/opencv.hpp>

namespace vio {

enum PnPMethod { ITERATIVE = 0, EPNP, P3P, DLS };

class PnPEstimator {
 public:
  PnPEstimator() {}

  static PnPEstimator *CreatePnPEstimator(PnPMethod method);
  static PnPEstimator *CreatePnPEstimatorOCV();

  virtual bool EstimatePose(const std::vector<cv::Point2f> &image_points,
                            const std::vector<cv::Point3f> &points3d,
                            const cv::Mat &K, std::vector<bool> &inliers,
                            cv::Mat &R_est, cv::Mat &t_est) = 0;

 protected:
};

}  // vio
