#include "feature_tracker.hpp"

#include <iostream>
#include <vector>

using namespace std;

int main() {
  cv::Mat image0 = cv::imread("/home/fan/Project/shumin_slam/feature_tracker/test_data/frame0.png");
  cv::Mat image1 = cv::imread("/home/fan/Project/shumin_slam/feature_tracker/test_data/frame1.png");

  if (!image0.data || !image1.data) {
    std::cerr << "Unable to load image.\n";
    return -1;
  } 

  FeatureTracker feature_tracker;

  vector<cv::KeyPoint> kp0, kp1;
  cv::Mat desc0, desc1;
  vector<cv::DMatch> matches;

  feature_tracker.DetectFeatureInFirstFrame(image0, kp0, desc0);
  feature_tracker.TrackFeature(desc0, image1, kp1, desc1, matches);

  std::cout << "Found " << matches.size() << " matches.\n"; 

  cv::Mat output_img;
  drawMatches(image0, kp0, image1, kp1, matches, output_img,
              cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
  
  cv::namedWindow( "result", cv::WINDOW_AUTOSIZE );
  cv::imshow("result", output_img); 
  cv::waitKey(0);

  return 0;
}
