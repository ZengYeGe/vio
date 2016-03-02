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
  vector<vector<cv::DMatch> > matches_pre_to_cur_k, matches_cur_to_pre_k;
  matcher_->knnMatch(pre_desc, cur_desc, matches_pre_to_cur_k, max_match_per_desc_);
  matcher_->knnMatch(cur_desc, pre_desc, matches_cur_to_pre_k, max_match_per_desc_);
 
  // Pick matches where the first one is much better than the second match.  
  vector<cv::DMatch> matches_pre_to_cur, matches_cur_to_pre;
  RatioTestFilter(matches_pre_to_cur_k, matches_pre_to_cur);
  RatioTestFilter(matches_cur_to_pre_k, matches_cur_to_pre);

  SymmetryTestFilter(matches_pre_to_cur, matches_cur_to_pre, matches);
  return true;
}

bool FeatureTracker::RatioTestFilter(vector<vector<cv::DMatch> > best_k,
                                     vector<cv::DMatch> &matches) {
  for (int i = 0; i < best_k.size(); ++i) {
    if (best_k[i][0].distance < nn_match_ratio_ * best_k[i][1].distance) {
      matches.push_back(best_k[i][0]);
    }
  }
  return true;
}

bool FeatureTracker::SymmetryTestFilter(const vector<cv::DMatch> &matches1,
                                        const vector<cv::DMatch> &matches2,
                                        vector<cv::DMatch> &final_matches) {
    final_matches.clear();
    for (vector<cv::DMatch>::const_iterator matchIterator1 = matches1.begin();
         matchIterator1 != matches1.end(); ++matchIterator1) {
        for (vector<cv::DMatch>::const_iterator matchIterator2 = matches2.begin();
             matchIterator2 != matches2.end();++matchIterator2) {
            if ((*matchIterator1).queryIdx == (*matchIterator2).trainIdx
               && (*matchIterator2).queryIdx ==(*matchIterator1).trainIdx) {
                final_matches.push_back(cv::DMatch((*matchIterator1).queryIdx,
                                                   (*matchIterator1).trainIdx,
                                                   (*matchIterator1).distance));
                break;
            }
        }
    }
} 


