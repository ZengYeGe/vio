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

  static MapInitializer *CreateMapInitializer(MapInitializerType type);
  static MapInitializer *CreateMapInitializerLIBMV();

  virtual bool Initialize(
      const std::vector<std::vector<cv::Vec2d> > &feature_vectors,
      const cv::Mat &K, std::vector<cv::Point3f> &points3d,
      std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts) = 0;

 private:
  Normalize(const std::vector<cv::Point2f> &points,
            std::vector<cv::Point2f> &norm_points,
            cv::Mat &p2norm_p);
};

}  // vio
