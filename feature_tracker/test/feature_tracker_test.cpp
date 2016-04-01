#include "feature_tracker.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

int TestTwoFrame();

int main(int argc, char** argv) {
  return TestTwoFrame();
}

int TestTwoFrame() {
  cv::Mat image0 = cv::imread("/home/fan/Project/shumin_slam/feature_tracker/test_data/frame0.png");
  cv::Mat image1 = cv::imread("/home/fan/Project/shumin_slam/feature_tracker/test_data/frame1.png");

  if (!image0.data || !image1.data) {
    std::cerr << "Unable to load image.\n";
    return -1;
  } 

  cv::Ptr<cv::Feature2D> detector = cv::ORB::create(10000);
  vio::FeatureTracker *feature_tracker = vio::FeatureTracker::CreateFeatureTracker(detector);
  vio::Frame new_frame(image0);
  feature_tracker->TrackFirstFrame(new_frame);
  vio::Frame second_frame(image1);
  std::vector<cv::DMatch> matches;
  feature_tracker->TrackFrame(new_frame, second_frame, matches);

  return 0;
}
