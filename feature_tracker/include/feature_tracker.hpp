#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

class FeatureTracker {
 public:
  // TODO: Make it base class, create instance based on chosen feature and matcher.
  FeatureTracker();
  ~FeatureTracker() {};

  // Used for first frame
  bool DetectFeatureInFirstFrame(const cv::Mat &first_frame, vector<cv::KeyPoint> &keypoints,
                                 cv::Mat &desc);
  // TODO: Pass in keypoint pos for fast search. e.g. search in a window.
  bool TrackFeature(cv::Mat &pre_desc, const cv::Mat &new_frame,
                    vector<cv::KeyPoint> &keypoints, 
                    cv::Mat &cur_desc, vector<cv::DMatch> &matches);
 private:
  cv::Ptr<cv::Feature2D> detector_;
  cv::Ptr<cv::DescriptorMatcher> matcher_;

  double nn_match_ratio_;
  // Count of best matches found per each query descriptor or less
  int max_match_per_desc_; 
  int max_num_keypoints_;
};
