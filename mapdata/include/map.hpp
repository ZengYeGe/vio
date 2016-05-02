#ifndef VIO_MAP_
#define VIO_MAP_

#include <memory>
#include <unordered_map>

#include <opencv2/opencv.hpp>

#include "keyframe.hpp"

namespace vio {

struct Landmark {
  cv::Point3f position;
};
/*
struct PoseEdge {
  cv::Mat R;
  cv::Mat t;
  int first_frame_index;
  int second_frame_index;
};
*/
struct FeatureMatchEdge {
  std::vector<cv::DMatch> matches;
  int first_frame_index;
  int second_frame_index;
};

class Map {
 public:
  enum MapState {
    WAIT_FOR_FIRSTFRAME = 0,
    WAIT_FOR_SECONDFRAME,
    WAIT_FOR_INIT,
    INITIALIZED,
  };

  Map();

  // TODO: Now assume new keyframe only match to last keyframe
  bool AddFirstKeyframe(std::unique_ptr<Keyframe> frame);
  bool AddNewKeyframeMatchToLastKeyframe(std::unique_ptr<Keyframe> frame,
                                         std::vector<cv::DMatch> &matches);

  /* ---------------- Initialization ----------------------------------------*/
  bool PrepareInitializationData(
      std::vector<std::vector<cv::Vec2d> > &feature_vectors);
  // This should be called after using PrepareInitializationData and initialized
  // point cloud.
  bool AddInitialization(const std::vector<cv::Point3f> &points3d,
                         const std::vector<bool> &points3d_mask,
                         const std::vector<cv::Mat> &Rs,
                         const std::vector<cv::Mat> &ts);

  /* ---------------- PnP Tracker ------------------------------------------*/
  bool PrepareEstimateLastFramePoseData(std::vector<cv::Point3f> &points3d,
                                        std::vector<cv::Point2f> &points2d,
                                        std::vector<int> &points_index);
  bool SetLastFramePose(const cv::Mat &R, const cv::Mat &t);
  bool AddPoseEdge(int first_frame_id, int second_frame_id, const cv::Mat &R,
                   const cv::Mat &t);

  /*-----------------New Lanmdarks-----------------------------------------*/
  bool PrepareUninitedPointsFromLastTwoFrames(std::vector<cv::Vec2d> &kp0,
                                              std::vector<cv::Vec2d> &kp1,
                                              FramePose &p0, FramePose &p1);
  bool AddInitedPoints(const std::vector<cv::Point3f> &points3d,
                       const std::vector<bool> &points3d_mask);

 /* ---------------Bundle adjustment-------------------------------------*/
  bool PrepareOptimization(std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts,
                           std::vector<cv::Point3f> &points,
                           std::vector<int> &obs_camera_idx,
                           std::vector<int> &obs_point_idx,
                           std::vector<cv::Vec2d> &obs_feature);

  bool ApplyOptimization(const std::vector<cv::Mat> &Rs,
                         const std::vector<cv::Mat> &ts,
                         const std::vector<cv::Point3f> &points);

  bool PrintStats();

  const Keyframe &GetLastKeyframe() const { return *(keyframes_.back()); }
  int num_frame() const { return keyframes_.size(); }

 private:
  bool AddCoordToUninitedPoints(const std::vector<cv::Point3f> &points3d,
                                const std::vector<bool> &points3d_mask);
  bool PruneShortTrackLandmarks();

  MapState map_state_;

  std::unordered_map<int, int> keyframe_id_to_index_;
  std::vector<std::unique_ptr<Keyframe> > keyframes_;

  // match_edges[i] is the match between keyframes_[i] and keyframes_[i + 1]
  // TODO: Not sure it's needed.
  std::vector<FeatureMatchEdge> match_edges_;

  std::vector<Landmark> landmarks_;

  // landmark_to_feature[i][j] is the no. of feature of ith landmark in
  // |landmarks_| in |keyframes_[j]|
  // The size should be [size of landmarks][size of keyframes]
  std::vector<std::unordered_map<int, int> > landmark_to_feature_;
  // Temporary landmarks that generated from feature matches
  std::vector<std::unordered_map<int, int> > uninited_landmark_to_feature_;

  // feature_to_landmark[i][j] is the no. of landmark of ith feature in
  // keyframe[j]
  // The size should be [size of keyframe][number of features in keyframe i]
  std::vector<std::unordered_map<int, int> > feature_to_landmark_;
};

}  // vio

#endif
