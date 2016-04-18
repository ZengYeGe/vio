#include <memory>

#include <opencv2/opencv.hpp>

#include "keyframe.hpp"

namespace vio {

struct Landmark {
  cv::Point3d position;
};

struct PoseEdge {
  cv::Mat transform;
};

class Map {
 public:
  Map();

  bool AddFirstKeyframe(std::unique_ptr<Keyframe> frame);

  const Keyframe &GetLastKeyframe() const;

 private:
  std::vector<std::unique_ptr<Keyframe> > keyframes_;
  // pose_edges[i] is the transformation from keyframes_[i] to keyframes_[i + 1]
  std::vector<PoseEdge> pose_edges_;
  // match_edges[i] is the match between keyframes_[i] and keyframes_[i + 1]
  std::vector<cv::DMatch> match_edges_;

  std::vector<Landmark> landmarks_;

  // landmark_to_feature[i][j] is the no. of feature of ith landmark in
  // |landmarks_| in |keyframes_[j]|
  // The size should be [size of landmarks][size of keyframes]
  std::vector<std::vector<int> > landmark_to_feature;
  // feature_to_landmark[i][j] is the no. of landmark of ith feature in
  // keyframe[j]
  // The size should be [size of keyframe][number of features in keyframe i]
  std::vector<std::vector<int> > feature_to_landmark;
};

}  // vio
