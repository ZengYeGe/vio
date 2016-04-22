#include "map.hpp"

#include <iostream>

namespace vio {

Map::Map()
  : map_state_(WAIT_FOR_FIRSTFRAME) {}

bool Map::AddFirstKeyframe(std::unique_ptr<Keyframe> frame) {
  if (map_state_ != WAIT_FOR_FIRSTFRAME) {
    std::cerr << "Error: First Keyframe already exists. Try reset map.\n";
    return false;
  }
  keyframe_id_to_index_[frame->frame_id()] = 0;
  keyframes_.push_back(std::move(frame));
  // Set pose of first frame
  cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat t = cv::Mat::ones(3, 1, CV_64F);
  keyframes_[0]->set_pose(R, t);

  map_state_ = WAIT_FOR_SECONDFRAME;
}

bool Map::AddNewKeyframeMatchToLastKeyframe(std::unique_ptr<Keyframe> frame,
                                            std::vector<cv::DMatch> &matches) {
  if (map_state_ == WAIT_FOR_FIRSTFRAME) {
    std::cerr << "Error: Missing first frame.\n";
    return false;
  }
  keyframe_id_to_index_[frame->frame_id()] = keyframes_.size();
  keyframes_.push_back(std::move(frame));

  // -------------- Add new match edge
  match_edges_.resize(match_edges_.size() + 1);
  match_edges_.back().first_frame_index = keyframes_.size() - 2;
  match_edges_.back().second_frame_index = keyframes_.size() - 1;
  const int l_index = match_edges_.back().first_frame_index;   // last frame index
  const int n_index = match_edges_.back().second_frame_index;  // new frame index

  feature_to_landmark_.resize(n_index + 1);
  for (int i = 0; i < matches.size(); ++i) {
    // Find existing landmark
    auto ld_id_ptr = feature_to_landmark_[l_index].find(matches[i].queryIdx);
    // If the feature is not a landmark yet
    if (ld_id_ptr == feature_to_landmark_[l_index].end()) {
      // Add a new landmark
      const int num_uninited_ld = uninited_landmark_to_feature_.size();
      uninited_landmark_to_feature_.resize(num_uninited_ld + 1);
      uninited_landmark_to_feature_.back()[n_index] = matches[i].trainIdx;
      uninited_landmark_to_feature_.back()[l_index] = matches[i].queryIdx;
    } else {
      int landmark_id = ld_id_ptr->second;
      // Link a feature to a landmark
      feature_to_landmark_[n_index][matches[i].trainIdx] = landmark_id;
      // Link a landmark to a feature in the new frame
      landmark_to_feature_[landmark_id][n_index] = matches[i].trainIdx;
    }
  }

  match_edges_.back().matches = std::move(matches);
  if (map_state_ == WAIT_FOR_SECONDFRAME) map_state_ = WAIT_FOR_INIT;

  return true;
}

bool Map::PrepareInitializationData(
    std::vector<std::vector<cv::Vec2d> > &feature_vectors) {
  if (map_state_ != WAIT_FOR_INIT) {
    std::cerr << "Error: Could not initialize.\n";
    return false;
  }
  std::cout << "Prepare initialization data for " << keyframes_.size()
            << " frames.\n";
  feature_vectors.resize(keyframes_.size());

  const int start_frame_id = 0;
  const int end_frame_id = keyframes_.size() - 1;
  const int num_landmark = uninited_landmark_to_feature_.size();

  // Initialize feature vector
  // TODO: Optimize, if the initialization cost much time.
  for (int frame_id = 0; frame_id < keyframes_.size(); ++frame_id) {
    feature_vectors[frame_id].resize(num_landmark, cv::Vec2d(-1, -1));
  }
  for (int ld_id = 0; ld_id < num_landmark; ++ld_id) {
    for (auto &ld_feature_id : uninited_landmark_to_feature_[ld_id]) {
      const cv::KeyPoint &kp = (keyframes_[ld_feature_id.first]
                                    ->image_frame()
                                    .keypoints())[ld_feature_id.second];
      feature_vectors[ld_feature_id.first][ld_id] = cv::Vec2d(kp.pt.x, kp.pt.y);
    }
  }
  return true;
}

bool Map::AddInitialization(const std::vector<cv::Point3f> &points3d,
                            const std::vector<bool> &points3d_mask,
                            const std::vector<cv::Mat> &Rs,
                            const std::vector<cv::Mat> &ts) {
  if (map_state_ != WAIT_FOR_INIT) {
    std::cerr << "Error: Could not add initialization.\n";
    return false;
  }
  if (Rs.size() != keyframes_.size() || ts.size() != keyframes_.size() ||
      points3d.size() != uninited_landmark_to_feature_.size() ||
      points3d_mask.size() != uninited_landmark_to_feature_.size()) {
    std::cerr << "Error: Initialization data doesn't match Map data.\n";
    return false;
  }

  // TODO: Refactor. Only add keyframe after initialization.

  // TODO: Use pprof to see where is the bottleneck
  // TODO: Try use GPU to replace the bottleneck
  landmark_to_feature_.clear();
  for (int ld_id = 0; ld_id < points3d.size(); ++ld_id) {
    if (points3d_mask[ld_id]) {
      landmark_to_feature_.push_back(
          std::move(uninited_landmark_to_feature_[ld_id]));
      Landmark new_ld;
      new_ld.position = points3d[ld_id];
      const int landmark_id = landmarks_.size();
      landmarks_.push_back(new_ld);

      for (auto &feature_in_frame : landmark_to_feature_.back()) {
        const int frame_id = feature_in_frame.first;
        const int feature_id = feature_in_frame.second;
        feature_to_landmark_[frame_id][feature_id] = landmark_id;
      }
    }
  }

  std::cout << "Inited points: " << landmarks_.size() << " / " 
            << uninited_landmark_to_feature_.size()
            << " landmarks.";
  uninited_landmark_to_feature_.clear();

  for (int i = 0; i < keyframes_.size(); ++i) {
    keyframes_[i]->set_pose(Rs[i], ts[i]);
    keyframes_[i]->set_pose_inited(true);
  }

  map_state_ = INITIALIZED;
  return true;
}

bool Map::PrepareEstimateLastFramePoseData(std::vector<cv::Point3f> &points3d,
                                           std::vector<cv::Point2f> &points2d,
                                           std::vector<int> &points_index) {
  if (map_state_ != INITIALIZED) {
    std::cerr << "Error: Map not initialized yet.\n";
    return false;
  }
  if (keyframes_.back()->pose_inited()) {
    std::cerr << "Error: Last frame already inited pose.\n";
    return false;
  }

  points3d.clear();
  points2d.clear();
  points_index.clear();

  const int l_index = match_edges_.back().first_frame_index;
  const int n_index = match_edges_.back().second_frame_index;
  const std::vector<cv::DMatch> &matches = match_edges_.back().matches;

  // TODO: This is duplicated when add a new keyframe
  for (int i = 0; i < matches.size(); ++i) {
    auto ld_id_ptr = feature_to_landmark_[n_index].find(matches[i].trainIdx);
    // If the feature is not a landmark yet
    if (ld_id_ptr != feature_to_landmark_[n_index].end()) {
      const int ft_id = ld_id_ptr->first;
      const int ld_id = ld_id_ptr->second;
      points_index.push_back(ft_id);
      points3d.push_back(landmarks_[ld_id].position);
      points2d.push_back(
        keyframes_.back()->image_frame().keypoints()[ft_id].pt);
    }
  }

  std::cout << "Found " << points_index.size() << " 2d to 3d match for pnp.\n";
  keyframes_.back()->set_pose_inited(true);

  return true;
}

bool Map::SetLastFramePose(const cv::Mat &R, const cv::Mat &t) {
  keyframes_.back()->set_pose(R, t);
};

}  // vio
