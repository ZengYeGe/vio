#ifndef VIO_MAP_
#define VIO_MAP_

#include <memory>
#include <unordered_map>

#include <opencv2/opencv.hpp>

#include "keyframe.hpp"

namespace vio {

struct Landmark {
  cv::Point3d position;
};

struct PoseEdge {
  cv::Mat transform;
  int first_frame_index;
  int second_frame_index;
};

struct FeatureMatchEdge {
  std::vector<cv::DMatch> matches;
  int first_frame_index;
  int second_frame_index;
};

class Map {
 public:
  Map();

  // TODO: Now assume new keyframe only match to last keyframe
  bool AddFirstKeyframe(std::unique_ptr<Keyframe> frame);
  bool AddNewKeyframeMatchToLastKeyframe(std::unique_ptr<Keyframe> frame,
                                         std::vector<cv::DMatch> &matches);

  /* ---------------- Initialization ----------------------------------------*/
  bool PrepareInitializationData(
      std::vector<std::vector<cv::Vec2d> > &feature_vectors);
  bool AddTwoFrameInitalizedLandmarks(int first_frame_id, int second_frame_id,
                                      const std::vector<cv::Point3d> &points3d,
                                      const std::vector<bool> &match_mask);
  /* ---------------- PnP Tracker ------------------------------------------*/
  bool PrepareEstimateLastFramePoseData();
  bool AddPoseEdge(int first_frame_id, int second_frame_id,
                   const cv::Mat &relative_pose);

  /* ----------------------------------------------------------------------*/
  // Remove not triangulated landmarks.
  void ClearRedundantLandmarks();

  // PrintStats

  const Keyframe &GetLastKeyframe() const { return *(keyframes_.back()); }

  int num_frame() const { return keyframes_.size(); }

 private:
  std::unordered_map<int, int> keyframe_id_to_index_;
  std::vector<std::unique_ptr<Keyframe> > keyframes_;
  // pose_edges[i] is the transformation from keyframes_[i] to keyframes_[i + 1]
  std::vector<PoseEdge> pose_edges_;
  // match_edges[i] is the match between keyframes_[i] and keyframes_[i + 1]
  // TODO: Not sure it's needed.
  std::vector<FeatureMatchEdge> match_edges_;

  std::vector<Landmark> landmarks_;
  bool landmarks_initialized_;

  // landmark_to_feature[i][j] is the no. of feature of ith landmark in
  // |landmarks_| in |keyframes_[j]|
  // The size should be [size of landmarks][size of keyframes]
  std::vector<std::unordered_map<int, int> > landmark_to_feature;
  // feature_to_landmark[i][j] is the no. of landmark of ith feature in
  // keyframe[j]
  // The size should be [size of keyframe][number of features in keyframe i]
  std::vector<std::unordered_map<int, int> > feature_to_landmark;
};

}  // vio

#endif
