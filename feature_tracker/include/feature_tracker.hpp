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
  bool TrackFeature(const vector<cv::KeyPoint> &pre_kp,
                    const cv::Mat &pre_desc, const cv::Mat &new_frame,
                    vector<cv::KeyPoint> &keypoints, 
                    cv::Mat &cur_desc, vector<cv::DMatch> &matches);
 private:
  bool RatioTestFilter(vector<vector<cv::DMatch> > best_k, vector<cv::DMatch> &matches);
  // TODO: Right now, it's O(n^2) search time.
  bool SymmetryTestFilter(const vector<cv::DMatch> &matches1,
                          const vector<cv::DMatch> &matches2,
                          vector<cv::DMatch> &final_matches);

  bool RemoveOutlierMatch(const vector<cv::KeyPoint> &pre_kp,
                          const vector<cv::KeyPoint> &cur_kp,
                          vector<cv::DMatch> &matches);

  cv::Ptr<cv::Feature2D> detector_;
  cv::Ptr<cv::DescriptorMatcher> matcher_;

  double nn_match_ratio_;
  // Count of best matches found per each query descriptor or less
  int max_match_per_desc_; 
  int max_num_keypoints_;
};
