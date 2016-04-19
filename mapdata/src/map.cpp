#include "map.hpp"

#include <iostream>

namespace vio {

Map::Map() : landmarks_initialized_(false) {}

bool Map::AddFirstKeyframe(std::unique_ptr<Keyframe> frame) {
  if (keyframes_.size() > 0) {
    std::cerr << "Error: First Keyframe already exists. Try reset map.\n";
    return false;
  }
  keyframe_id_to_index_[frame->frame_id()] = 0;
  keyframes_.push_back(std::move(frame));
}

bool Map::AddNewKeyframeMatchToLastKeyframe(std::unique_ptr<Keyframe> frame,
                                            std::vector<cv::DMatch> &matches) {
  if (keyframes_.size() == 0) {
    std::cerr << "Error: Map has no keyframe yet.\n";
    return false;
  }
  keyframe_id_to_index_[frame->frame_id()] = keyframes_.size();
  keyframes_.push_back(std::move(frame));

  // -------------- Add new match edge
  FeatureMatchEdge match_edge;
  match_edge.first_frame_index = keyframes_.size() - 2;
  match_edge.second_frame_index = keyframes_.size() - 1;
  match_edge.matches = std::move(matches);

  if (!landmarks_initialized_) {
    std::cout << "Added a new frame without landmarks initialized.\n";
    return true;
  }

  const int l_index = match_edge.first_frame_index;   // last frame index
  const int n_index = match_edge.second_frame_index;  // new frame index
  feature_to_landmark.resize(feature_to_landmark.size() + 1);

  if (landmarks_initialized_) {
    for (int i = 0; i < matches.size(); ++i) {
      // Find existing landmark
      auto ld_id_ptr = feature_to_landmark[l_index].find(matches[i].queryIdx);
      // If the feature is not a landmark yet
      if (ld_id_ptr == feature_to_landmark[l_index].end()) continue;
      int landmark_id = ld_id_ptr->second;

      // Link a feature to a landmark
      feature_to_landmark[n_index][matches[i].trainIdx] = landmark_id;
      // Link a landmark to a feature in the new frame
      landmark_to_feature[landmark_id][n_index] = matches[i].trainIdx;
    }
  }
  return true;
}

}  // vio
