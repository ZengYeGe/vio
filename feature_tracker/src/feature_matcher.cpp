#include "feature_matcher.hpp"

namespace vio {

bool FeatureMatcher::SymmetryTestFilter(
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

bool FeatureMatcher::RatioTestFilter(
    std::vector<std::vector<cv::DMatch> > best_k,
    std::vector<cv::DMatch> &matches) {
  for (int i = 0; i < best_k.size(); ++i) {
    if (best_k[i][0].distance < nn_match_ratio_ * best_k[i][1].distance) {
      matches.push_back(best_k[i][0]);
    }
  }
  return true;
}


} // vio
