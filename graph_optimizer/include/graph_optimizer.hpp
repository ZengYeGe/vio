#include "opencv2/opencv.hpp"

namespace vio {

enum GraphOptimizerMethod { CERES, G2O };

class GraphOptimizer {
 public:
  GraphOptimizer();

  static GraphOptimizer *CreateGraphOptimizer(GraphOptimizerMethod method);
  static GraphOptimizer *CreateGraphOptimizerCeres();

  virtual bool Optimize(const std::vector<cv::Mat> &K, std::vector<cv::Mat> &Rs,
                        std::vector<cv::Mat> &ts,
                        std::vector<cv::Point3f> &points,
                        const std::vector<int> &obs_camera_idx,
                        const std::vector<int> &obs_point_idx,
                        const std::vector<cv::Vec2d> &obs_feature) = 0;
};

}  // vio
