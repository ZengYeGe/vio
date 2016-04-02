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
  cv::Mat image0 = cv::imread("../feature_tracker/test_data/frame0.png");
  cv::Mat image1 = cv::imread("../feature_tracker/test_data/frame1.png");

  if (!image0.data || !image1.data) {
    std::cerr << "Unable to load image.\n";
    return -1;
  } 

  cv::Ptr<cv::Feature2D> detector = cv::ORB::create(10000);
  vio::FeatureTracker *feature_tracker = vio::FeatureTracker::CreateFeatureTracker(detector);
  vio::Frame first_frame(image0);
  feature_tracker->TrackFirstFrame(first_frame);
  vio::Frame second_frame(image1);
  std::vector<cv::DMatch> matches;
  feature_tracker->TrackFrame(first_frame, second_frame, matches);

  cv::Mat output_img;
  drawMatches(first_frame.GetImage(), first_frame.GetFeatures().keypoints,
              second_frame.GetImage(), second_frame.GetFeatures().keypoints,
              matches, output_img,
              cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
  
  cv::namedWindow( "result", cv::WINDOW_AUTOSIZE );
  cv::imshow("result", output_img); 
  cv::waitKey(0);

  return 0;
}
