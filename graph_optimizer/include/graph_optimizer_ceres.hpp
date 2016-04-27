#include "graph_optimizer.hpp"

#include <iostream>
#include <vector>

#include "ceres/ceres.h"

#include "reprojection_error.hpp"

namespace vio {

class GraphOptimizerCeres : public GraphOptimizer {
 public:
  GraphOptimizerCeres() {}

  bool Optimize(const std::vector<cv::Mat> &K, std::vector<cv::Mat> &Rs,
                std::vector<cv::Mat> &ts, std::vector<cv::Point3f> &points,
                const std::vector<int> &obs_camera_idx,
                const std::vector<int> &obs_point_idx,
                const std::vector<cv::Vec2d> &obs_feature) override;

 private:
  bool ConstructProblem(const std::vector<cv::Mat> &Rs,
                        const std::vector<cv::Mat> &ts,
                        const std::vector<cv::Point3f> &points);

  bool AssignOptimizedResult(std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts,
                             std::vector<cv::Point3f> &points);
  // size is 9 * num_cameras
  std::vector<double> cameras_;
  // size is 3 * num_points
  std::vector<double> points_;
};

}  // vio
