#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <vector>

using namespace std;

class LandmarkServer {
 public:
  LandmarkServer();
  ~LandmarkServer();

  bool AddFirstFrameFeature(const vector<cv::KeyPoint> &kp);
  // For now, camera_pose_id0 has to be already a frame in the server.
  // kp1 is last_frame matches to new_frame
  bool AddNewFeatureAssociationToLastFrame(const vector<cv::KeyPoint> &kp1, 
                                           const vector<cv::DMatch> &matches);

  bool MakeFeatureVectorsForReconstruct(vector<vector<cv::Vec2d> > &feature_vectors);

  bool PrintStats();

 private:
  // Number of frame added
  int num_frame_;
  // Number of landmarks
  int num_landmark_;
  // 3D position of each landmark
  vector<cv::Point3d> landmark_pos_;
  // Each 3D landmark correspond to a feature in camera
  // ith landmark is number landmark_in_camera_id[i][j] feature in jth camera
  vector<vector<int> > landmark_in_camera_id_;
  // Each feature in a camera correspond to a landmark
  // feature_in_landmark_id[i][j] :
  // jth feature in ith camera is number feature_in_landmark_id[i][j] landmark
  vector<vector<int> > feature_in_landmark_id_;
  // (x, y) of feature ith in jth camera.
  vector<vector<cv::Vec2d> > feature_pos_in_camera_;

};
