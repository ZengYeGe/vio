#include <opencv2/opencv.hpp>

#include <vector>

using namespace std;

class KeyframeSelector {
 public:
  KeyframeSelector() : num_matches_thres_(600) {}
  ~KeyframeSelector() {}

  bool isKeyframe(const vector<cv::DMatch> &matches) {
    if (matches.size() < num_matches_thres_) return true;
    return false;
  }

  bool isKeyframe(const cv::Mat &img, const vector<cv::KeyPoint> &kp,
                  const vector<cv::DMatch> &matches) {
    if (matches.size() < num_matches_thres_) return true;
    return false;
  }

 private:
  int num_matches_thres_;
};
