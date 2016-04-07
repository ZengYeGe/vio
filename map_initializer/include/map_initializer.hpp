#include <vector>

#include <opencv2/opencv.hpp>

namespace vio {

// TODO: Implement each of them.
enum MapInitializerType {
  LIVMV,
  ORBSLAM,
  PTAM,
  NORMALIZED8POINTHOMOGRAPHY,
  NORMALIZED8POINTFUNDAMENTAL,
  NORMALIZED8POINTFUNDAMENTALRANSAC,
  FIVEPOINT,
  FIVEPOINTEASY
};

class MapInitializer {
 public:
  MapInitializer() {}
  ~MapInitializer() {}

  static MapInitializer *CreateMapInitializer(MapInitializerType type);
  static MapInitializer *CreateMapInitializerLIBMV();
  static MapInitializer *CreateMapInitializer8Point();

  virtual bool Initialize(
      const std::vector<std::vector<cv::Vec2d> > &feature_vectors,
      const cv::Mat &K, std::vector<cv::Point3f> &points3d,
      std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts) = 0;

 protected:
  void Normalize(const std::vector<cv::Vec2d> &points,
                 std::vector<cv::Vec2d> &norm_points, cv::Mat &p2norm_p);
};

}  // vio
