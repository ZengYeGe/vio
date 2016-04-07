#include "map_initializer.hpp"

#include <opencv2/sfm.hpp>

namespace vio {

class MapInitializer8Point : public MapInitializer {
 public:
  MapInitializer8Point() {}
  ~MapInitializer8Point() {}

  virtual bool Initialize(
      const std::vector<std::vector<cv::Vec2d> > &feature_vectors,
      const cv::Mat &K, std::vector<cv::Point3f> &points3d,
      std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts) override;

 protected:
  bool InitializeTwoFrames(const std::vector<cv::Vec2d> &kp0,
                           const std::vector<cv::Vec2d> &kp1, const cv::Mat &K,
                           std::vector<cv::Point3f> &points3d,
                           std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts);

  // Handle n points, at least 8.
  bool ComputeF(const std::vector<cv::Vec2d> &kp0,
                const std::vector<cv::Vec2d> &kp1, cv::Mat &F);
  cv::Mat ComputeFOpenCV(const std::vector<cv::Vec2d> &kp0,
                         const std::vector<cv::Vec2d> &kp1);

  // cv::Mat ComputeEfromF(const cv::Mat &F, const cv::Mat &K);
  void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
  void Triangulate(const cv::Vec2d &kp1, const cv::Vec2d &kp2,
                   const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
  int EvaluateSolutionRT(const cv::Mat &R, const cv::Mat &t,
                         const std::vector<cv::Vec2d> &kp0,
                         const std::vector<cv::Vec2d> &kp1,
                         const std::vector<bool> &match_inliers,
                         const cv::Mat &K, std::vector<cv::Point3f> &points_3d,
                         double th2, std::vector<bool> &vbGood,
                         double &parallax);
};

}  // vio
