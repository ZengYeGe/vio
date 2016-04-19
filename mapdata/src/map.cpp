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

  const int l_index = match_edge.first_frame_index;   // last frame index
  const int n_index = match_edge.second_frame_index;  // new frame index
  feature_to_landmark.resize(n_index + 1);

  for (int i = 0; i < matches.size(); ++i) {
      // Find existing landmark
      auto ld_id_ptr = feature_to_landmark[l_index].find(matches[i].queryIdx);
      // If the feature is not a landmark yet
      if (ld_id_ptr == feature_to_landmark[l_index].end()) {
        // Add a new landmark
        const int landmark_id = landmark_to_feature.size();
        landmark_to_feature.resize(landmark_id + 1);
        landmark_to_feature.back()[n_index] = matches[i].trainIdx;
        landmark_to_feature.back()[l_index] = matches[i].queryIdx;
        feature_to_landmark[l_index][matches[i].queryIdx] = landmark_id;
        feature_to_landmark[n_index][matches[i].trainIdx] = landmark_id;
      } else {
        int landmark_id = ld_id_ptr->second;
        // Link a feature to a landmark
        feature_to_landmark[n_index][matches[i].trainIdx] = landmark_id;
        // Link a landmark to a feature in the new frame
        landmark_to_feature[landmark_id][n_index] = matches[i].trainIdx;
      }
  }
  match_edge.matches = std::move(matches);
 
  return true;
}

// TODO: Unfinished
bool Map::PrepareInitializationData(
     std::vector<std::vector<cv::Vec2d> > &feature_vectors) {
  if (keyframes_.size() < 2) {
    std::cerr << "Error: Not enough keyframes for initialization.\n";
    return false;
  }
  std::cout << "Prepare initialization data for " << keyframes_.size() << " frames.\n";
  feature_vectors.resize(keyframes_.size());

  const int start_frame_id = 0;
  const int end_frame_id = keyframes_.size() - 1;
  const int num_landmark = landmark_to_feature.size();
  
  // Initialize feature vector
  // TODO: Optimize, if the initialization cost much time.
  for (int frame_id = 0; frame_id < keyframes_.size(); ++frame_id) {
    feature_vectors[frame_id].resize(num_landmark, cv::Vec2d(-1, -1));
  }
  for (int ld_id = 0; ld_id < num_landmark; ++ld_id) {
    for (auto &ld_feature_id : landmark_to_feature[ld_id]) {
      const cv::KeyPoint &kp =
          (keyframes_[ld_feature_id.first]->image_frame().keypoints())[ld_feature_id.second];
      feature_vectors[ld_feature_id.first][ld_id] = cv::Vec2d(kp.pt.x, kp.pt.y);
    }
  }
  return true;
}

}  // vio
