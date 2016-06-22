#include <vector>

#include <opencv2/opencv.hpp>

#include "multiview.hpp"
// #include "../../multiview_helper/include/multiview.hpp"

namespace vio {

// TODO: Implement each of them.
enum MapInitializerMethod {
  LIVMV,
  PTAM,
  NORMALIZED8POINTFUNDAMENTAL,
  ORBSLAM_F_OR_H,
  FIVEPOINT,
  FIVEPOINTEASY
};

struct MapInitializerOptions {
  MapInitializerOptions()
      : method(NORMALIZED8POINTFUNDAMENTAL),
        use_f_ransac(false),
        f_ransac_confidence(0.99),
        f_ransac_max_dist_to_epipolar(2),
        reprojection_error_thres(5),
        verbose(false) {}

  MapInitializerMethod method;
  // Compute Fundamental
  bool use_f_ransac;
  double f_ransac_confidence;
  double f_ransac_max_dist_to_epipolar;

  // triangulation
  double reprojection_error_thres;

  bool verbose;

  void read(const cv::FileNode &node) {
    method = static_cast<MapInitializerMethod>((int)node["Method"]);
    use_f_ransac = (int)node["F_USE_RANSAC"];
    f_ransac_confidence = (double)node["F_RANSAC_CONFIDENCE"];
    f_ransac_max_dist_to_epipolar = (double)node["F_RANSAC_MAX_DIST"];
    reprojection_error_thres = (double)node["REPROJECTION_ERROR_THRESHOLD"];
    verbose = (int)node["VERBOSE"];
  }
};

// Following must be defined for the serialization in FileStorage to work
static void read(
    const cv::FileNode &node, MapInitializerOptions &x,
    const MapInitializerOptions &default_value = MapInitializerOptions()) {
  if (node.empty())
    x = default_value;
  else
    x.read(node);
}

class MapInitializer {
 public:
  MapInitializer() {}
  MapInitializer(MapInitializerOptions option) : option_(option) {}
  ~MapInitializer() {}

  static MapInitializer *CreateMapInitializer(MapInitializerOptions option);

  static MapInitializer *CreateMapInitializerLIBMV();
  static MapInitializer *CreateMapInitializer8Point(
      MapInitializerOptions option);
  static MapInitializer *CreateMapInitializerORBSLAM(
      MapInitializerOptions option);

  virtual bool Initialize(
      const std::vector<std::vector<cv::Vec2d> > &feature_vectors,
      const cv::Mat &K, std::vector<cv::Point3f> &points3d,
      std::vector<bool> &points3d_mask, std::vector<cv::Mat> &Rs,
      std::vector<cv::Mat> &ts) = 0;

 protected:
  MapInitializerOptions option_;
};

}  // vio
