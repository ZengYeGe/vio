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

  bool initialize3DPointsFromViews(
      const vector<vector<cv::Vec2d> > &feature_vectors,
      const cv::Matx33d &inital_camera_matrix, vector<cv::Mat> &points3d_ests,
      vector<cv::Mat> &pose_ests, cv::Mat &refined_camera_matrix);

 private:
  bool is_projective_;
  bool verbose_;
};
