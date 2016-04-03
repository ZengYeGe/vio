#include <vector>

#include <opencv2/opencv.hpp>

namespace vio {

// TODO: Implement each of them.
enum MapInitializerType {
  LIVMV,
  ORBSLAM,
  PTAM,
  NORMALIZED8POINTHOMOGRAPHY,
  NORMALIZED8POINTFUNDMENTAL,
  FIVEPOINT,
  FIVEPOINTEASY
};  

class MapInitializer {
 public:
  MapInitializer() {}
  ~MapInitializer() {}

  static MapInitializer *CreateMapInitializer(MapInitializerType type) {
    switch (type) {
      case LIVMV:
        return CreateMapInitializerLIBMV();
      default:
        return nullptr;
    }
  }
  static MapInitializer *CreateMapInitializerLIBMV();

  virtual bool Initialize(const std::vector<std::vector<cv::Vec2d> > &feature_vectors,
                     const cv::Mat &K, std::vector<cv::Point3f> &points3d,
                     std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts) = 0;
};

} // vio
