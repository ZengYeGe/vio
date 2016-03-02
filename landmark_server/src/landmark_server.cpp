#include "landmark_server.hpp"

using namespace std;

LandmarkServer::LandmarkServer()
    : num_frame_(0),
      num_landmark_(0) {}

LandmarkServer::~LandmarkServer() {}

bool LandmarkServer::AddFirstFrameFeature(const vector<cv::KeyPoint> &kp) {
  // Add new camera frame
  vector<cv::Vec2d> feature_pos(kp.size());
  for (int i = 0; i < kp.size(); ++i) {
    feature_pos[i][0] = kp[i].pt.x;
    feature_pos[i][1] = kp[i].pt.y;
  }
  feature_pos_in_camera_.push_back(feature_pos);

  vector<int> first_frame_feature_to_landmark(kp.size(), -1);
  feature_in_landmark_id_.push_back(first_frame_feature_to_landmark);

  num_frame_++;

  return true;
}
bool LandmarkServer::AddNewFeatureAssociationToLastFrame(const vector<cv::KeyPoint> &kp1,
                                                         const vector<cv::DMatch> &matches) {
  // Add new camera frame
  vector<cv::Vec2d> feature_pos(kp1.size());
  for (int i = 0; i < kp1.size(); ++i) {
    feature_pos[i][0] = kp1[i].pt.x;
    feature_pos[i][1] = kp1[i].pt.y;
  }
  feature_pos_in_camera_.push_back(feature_pos);
  
  // Each landmark need to add a reference to the new frame
  for (int i = 0; i < landmark_in_camera_id_.size(); ++i)
    landmark_in_camera_id_[i].push_back(-1);

  // Find if matched features already in landmark list
  vector<int> new_feature_in_landmark_id(kp1.size(), -1);
  for (int i = 0; i < matches.size(); ++i) {
    int pre_id = matches[i].queryIdx;
    int new_id = matches[i].trainIdx;
    if (feature_in_landmark_id_.back()[pre_id] != -1) {
      // map a feature in the latest frame to a landmark
      new_feature_in_landmark_id[new_id] = feature_in_landmark_id_.back()[pre_id];
      // map landmark to a feature in the latest frame
      landmark_in_camera_id_[new_feature_in_landmark_id[new_id]].back() = new_id;
    } else {
      // The feature is not in the landmark set, add 
      vector<int> new_landmark_to_feature(feature_pos_in_camera_.size() - 2, -1);
      // map the landmark to new frame and last frame
      new_landmark_to_feature.push_back(pre_id);
      new_landmark_to_feature.push_back(new_id);
      landmark_in_camera_id_.push_back(new_landmark_to_feature);
      // map feature to the newly added landmark
      feature_in_landmark_id_.back()[pre_id] = landmark_in_camera_id_.size();
      new_feature_in_landmark_id[new_id] = landmark_in_camera_id_.size();
    }
  }
  feature_in_landmark_id_.push_back(new_feature_in_landmark_id);

  num_frame_++; 
  num_landmark_ = landmark_in_camera_id_.size();
  return true;
}

bool LandmarkServer::MakeFeatureVectorsForReconstruct(vector<vector<cv::Vec2d> > &feature_vectors) {
  feature_vectors.resize(num_frame_);
  for (int frame_id = 0; frame_id < num_frame_; ++frame_id) {
    for (int landmark_id = 0; landmark_id < num_landmark_; ++landmark_id) {
      if (landmark_in_camera_id_[landmark_id][frame_id] != -1) {
        int feature_id = landmark_in_camera_id_[landmark_id][frame_id];
        feature_vectors[frame_id].push_back(feature_pos_in_camera_[frame_id][feature_id]);
      } else {
        feature_vectors[frame_id].push_back(cv::Vec2d(-1, -1));
      }
    }
  }
  return true;
}

bool LandmarkServer::PrintStats() {
  // Count each landmark is seen in how many frames.
  vector<int> seen_count(num_landmark_, 0);
  for (int i = 0; i < num_landmark_; ++i) {
    for (int j = 0; j < num_frame_; ++j)
      if (landmark_in_camera_id_[i][j] != -1)
        seen_count[i]++;
  }
  for (int count = num_frame_; count >= 1; --count) {
    int temp_count = 0;
    for (int i = 0; i < num_landmark_; ++i)
      if (seen_count[i] == count)
        temp_count++;
    cout << "Landmark seen " << count << " times: " << temp_count << endl;
  }
  return true; 
}


