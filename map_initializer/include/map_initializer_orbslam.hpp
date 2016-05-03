#include "map_initializer_8point.hpp"

#include <opencv2/sfm.hpp>

namespace vio {

class MapInitializerORBSLAM : public MapInitializer8Point {
 public:
  MapInitializerORBSLAM(MapInitializerOptions option)
      : MapInitializer8Point(option) {}
  ~MapInitializerORBSLAM() {}

  virtual bool Initialize(
      const std::vector<std::vector<cv::Vec2d> > &feature_vectors,
      const cv::Mat &K, std::vector<cv::Point3f> &points3d,
      std::vector<bool> &points3d_mask, std::vector<cv::Mat> &Rs,
      std::vector<cv::Mat> &ts) override;

};

} // vio
