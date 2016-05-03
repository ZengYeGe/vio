#include "map_initializer.hpp"

#include <opencv2/sfm.hpp>

namespace vio {

class MapInitializer7Point : public MapInitializer {
 public:
  MapInitializer7Point() {}
  ~MapInitializer7Point() {}

  virtual bool Initialize(
      const std::vector<std::vector<cv::Vec2d> > &feature_vectors,
      const cv::Mat &K, std::vector<cv::Point3f> &points3d,
      std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts) override;

 protected:
  bool InitializeTwoFrames(const std::vector<cv::Vec2d> &kp0,
                           const std::vector<cv::Vec2d> &kp1, const cv::Mat &K,
                           std::vector<cv::Point3f> &points3d,
                           std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts);
};

}  // vio
