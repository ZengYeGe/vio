#include <opencv2/opencv.hpp>
#include <opencv2/sfm.hpp>

#include <vector>

using namespace std;

// TODO: Make it a base class, use other method to initialize points.
// e.g. five points, loosely, tightly coupled.
class PoseOptimizer {
 public:
  PoseOptimizer() : is_projective_(true) {}
  PoseOptimizer(bool is_proj);
  ~PoseOptimizer(){};

  // Point matches in several frames.
  // position value is > 0.
  // -1 is not seen in the frame.
  bool initialize3DPointsFromViews(
      int num_frame, const vector<vector<cv::Vec2d> > &feature_vectors,
      const cv::Matx33d &inital_camera_matrix, vector<cv::Mat> &points3d_ests,
      vector<cv::Mat> &R_ests, vector<cv::Mat> &t_ests,
      cv::Matx33d &refined_camera_matrix);

  bool initialize3DPointsFromTwoViews(const vector<cv::KeyPoint> &points0,
                                      const vector<cv::KeyPoint> &points1,
                                      const cv::Matx33d &inital_camera_matrix,
                                      vector<cv::Point3f> &points3d_ests,
                                      cv::Mat &R, cv::Mat &t);

 private:
  bool reconstructTwoViews(const vector<bool> &match_inliers,
                           const vector<cv::KeyPoint> &pts0,
                           const vector<cv::KeyPoint> &pts1, const cv::Mat &F,
                           const cv::Mat &K, cv::Mat &R, cv::Mat &t,
                           vector<cv::Point3f> &points_3d,
                           vector<bool> &points_3d_valid, float minParallax,
                           int minTriangulated);
  int CheckRT(const cv::Mat &R, const cv::Mat &t,
              const vector<cv::KeyPoint> &kp0, const vector<cv::KeyPoint> &kp1,
              const vector<bool> &match_inliers, const cv::Mat &K,
              vector<cv::Point3f> &points_3d, float th2, vector<bool> &vbGood,
              float &parallax);

  void triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2,
                   const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
  void decomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
  void normalize(const vector<cv::Point2f> &points,
                 vector<cv::Point2f> &normalized_points, cv::Mat &p2norm_p);

  bool is_projective_;
  bool verbose_;

  int max_ransac_iterations_;
};
