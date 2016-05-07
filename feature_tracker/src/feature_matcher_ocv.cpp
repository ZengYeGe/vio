#include "feature_matcher_ocv.hpp"

#include <iostream>

#include "../../util/include/timer.hpp"

namespace vio {

bool FeatureMatcherOCV::Match(const ImageFrame &frame0, const ImageFrame &frame1,
                     std::vector<cv::DMatch> &matches) {
  const std::vector<cv::KeyPoint> &kp0 = frame0.keypoints();
  const std::vector<cv::KeyPoint> &kp1 = frame1.keypoints();
  const cv::Mat &desc0 = frame0.descriptors();
  const cv::Mat &desc1 = frame1.descriptors();

  Timer timer;
  timer.Start();

  std::vector<std::vector<cv::DMatch> > matches_0to1_k, matches_1to0_k;
  matcher_->knnMatch(desc0, desc1, matches_0to1_k, max_match_per_desc_);
  matcher_->knnMatch(desc1, desc0, matches_1to0_k, max_match_per_desc_);

//  matcher_->radiusMatch(desc0, desc1, matches_0to1_k, 20);
//  matcher_->radiusMatch(desc1, desc0, matches_1to0_k, 20);


  timer.Stop();
  std::cout << "Knn match time used: " << timer.GetInMs() << "ms.\n";
  timer.Start();

  // Pick matches where the first one is much better than the second match.
  std::vector<cv::DMatch> matches_0to1, matches_1to0;
  RatioTestFilter(matches_0to1_k, matches_0to1);
  RatioTestFilter(matches_1to0_k, matches_1to0);

  timer.Stop();
  std::cout << "Ratio test time used: " << timer.GetInMs() << "ms.\n";
  timer.Start();

  if (matches_0to1.size() < 10 || matches_1to0.size() < 10) {
    std::cerr << "Error: Not enough matches after ratio test.\n";
    std::cerr << "Match 0 to 1: " << matches_0to1.size() << " / "
              << matches_0to1_k.size() << std::endl;
    std::cerr << "Match 1 to 0: " << matches_1to0.size() << " / "
              << matches_1to0_k.size() << std::endl;
    return false;
  }
  // matches is pre to cur
  SymmetryTestFilter(matches_0to1, matches_1to0, matches);
  if (matches.size() < 5) {
    std::cerr << "Error: Not enough matches after symmetry test.\n";
    return false;
  }

  timer.Stop();
  std::cout << "Symmetry test time used: " << timer.GetInMs() << "ms.\n";
  timer.Start();

  RemoveOutlierMatch(kp0, kp1, matches);

  timer.Stop();
  std::cout << "F matrix outlier test time used: " << timer.GetInMs() << "ms.\n";

  if (matches.size() < 3) {
    std::cerr << "Error: Not enough matches after outlier removal.\n";
    return false;
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
      pre_matched_kp, cur_matched_kp, CV_FM_RANSAC, 0.5, 0.999, mask);
  int num_outlier = 0;
  std::vector<cv::DMatch> new_matches;
  for (int i = 0; i < matches.size(); ++i) {
    if ((unsigned int)mask.at<uchar>(i)) new_matches.push_back(matches[i]);
  }
  std::cout << "Outlier matches: " << matches.size() - new_matches.size()
            << std::endl;

  matches = std::move(new_matches);
  return true;
}

}  // vio
