#include "feature_tracker.hpp"

FeatureTracker::FeatureTracker()
    : nn_match_ratio_(0.8f),
      max_match_per_desc_(2),
      max_num_keypoints_(10000) {
  detector_ = cv::ORB::create();
  // TODO: Set orb max features.
  // detector_->setMaxFeatures(max_num_keypoints_);
  matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");
}

bool FeatureTracker::DetectFeatureInFirstFrame(const cv::Mat &first_frame,
                                               vector<cv::KeyPoint> &keypoints,
                                               cv::Mat &desc) {
   // noArray() is image mask for specifying where to find features.
   detector_->detectAndCompute(first_frame, cv::noArray(), keypoints, desc);
   return true;
}

bool FeatureTracker::TrackFeature(cv::Mat &pre_desc, const cv::Mat &new_frame,
                                  vector<cv::KeyPoint> &keypoints,
                                  cv::Mat &cur_desc, vector<cv::DMatch> &matches) {
  detector_->detectAndCompute(new_frame, cv::noArray(), keypoints, cur_desc);

  matches.clear();
  vector<vector<cv::DMatch> > best_k_matches;
  matcher_->knnMatch(pre_desc, cur_desc, best_k_matches, max_match_per_desc_);
  // Pick matches where the first one is much better than the second match.
  for (unsigned i = 0; i < best_k_matches.size(); ++i) {
    if(best_k_matches[i][0].distance < nn_match_ratio_ * best_k_matches[i][1].distance) {
      matches.push_back(best_k_matches[i][0]);
    }
  }
  return true;
}
