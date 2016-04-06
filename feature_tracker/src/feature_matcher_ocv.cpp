#include "feature_matcher_ocv.hpp"

#include <iostream>

namespace vio {

bool FeatureMatcherOCV::Match(const FeatureSet &features0,
                              const FeatureSet &features1,
                              std::vector<cv::DMatch> &matches) {
  std::vector<std::vector<cv::DMatch> > matches_0to1_k, matches_1to0_k;
  matcher_->knnMatch(features0.descriptors, features1.descriptors,
                     matches_0to1_k, max_match_per_desc_);
  matcher_->knnMatch(features1.descriptors, features0.descriptors,
                     matches_1to0_k, max_match_per_desc_);

  // Pick matches where the first one is much better than the second match.
  std::vector<cv::DMatch> matches_0to1, matches_1to0;
  RatioTestFilter(matches_0to1_k, matches_0to1);
  RatioTestFilter(matches_1to0_k, matches_1to0);

  // matches is pre to cur
  SymmetryTestFilter(matches_0to1, matches_1to0, matches);
  RemoveOutlierMatch(features0.keypoints, features1.keypoints, matches);

  return true;
}

bool FeatureMatcherOCV::RatioTestFilter(
    std::vector<std::vector<cv::DMatch> > best_k,
    std::vector<cv::DMatch> &matches) {
  for (int i = 0; i < best_k.size(); ++i) {
    if (best_k[i][0].distance < nn_match_ratio_ * best_k[i][1].distance) {
      matches.push_back(best_k[i][0]);
    }
  }
  return true;
}

bool FeatureMatcherOCV::SymmetryTestFilter(
    const std::vector<cv::DMatch> &matches1,
    const std::vector<cv::DMatch> &matches2,
    std::vector<cv::DMatch> &final_matches) {
  final_matches.clear();
  for (std::vector<cv::DMatch>::const_iterator matchIterator1 =
           matches1.begin();
       matchIterator1 != matches1.end(); ++matchIterator1) {
    for (std::vector<cv::DMatch>::const_iterator matchIterator2 =
             matches2.begin();
         matchIterator2 != matches2.end(); ++matchIterator2) {
      if ((*matchIterator1).queryIdx == (*matchIterator2).trainIdx &&
          (*matchIterator2).queryIdx == (*matchIterator1).trainIdx) {
        final_matches.push_back(cv::DMatch((*matchIterator1).queryIdx,
                                           (*matchIterator1).trainIdx,
                                           (*matchIterator1).distance));
        break;
      }
    }
  }
}

bool FeatureMatcherOCV::RemoveOutlierMatch(
    const std::vector<cv::KeyPoint> &pre_kp,
    const std::vector<cv::KeyPoint> &cur_kp, std::vector<cv::DMatch> &matches) {
  // TODO: Check the speed.
  std::vector<cv::Point2f> pre_matched_kp, cur_matched_kp;
  for (int i = 0; i < matches.size(); ++i) {
    pre_matched_kp.push_back(pre_kp[matches[i].queryIdx].pt);
    cur_matched_kp.push_back(cur_kp[matches[i].trainIdx].pt);
  }
  cv::Mat mask;
  // TODO: Need to tune the parameters, e.g. 3
  // TODO: Normalize
  cv::Mat fundamental_matrix = cv::findFundamentalMat(
      pre_matched_kp, cur_matched_kp, CV_FM_RANSAC, 0.2, 0.999, mask);
  int num_outlier = 0;
  std::vector<cv::DMatch> new_matches;
  for (int i = 0; i < matches.size(); ++i) {
    if ((unsigned int)mask.at<uchar>(i)) new_matches.push_back(matches[i]);
  }
  std::cout << "outlier matches: " << matches.size() - new_matches.size()
            << std::endl;

  matches = std::move(new_matches);
  return true;
}

}  // vio
