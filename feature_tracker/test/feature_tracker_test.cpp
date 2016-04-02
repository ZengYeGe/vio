#include "feature_tracker.hpp"
#include "keyframe_selector.hpp"

#ifdef __linux__
#include <dirent.h>
#endif

#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// TODO: Add util folder to include
#include "../../util/include/util.hpp"

using namespace std;

struct Options {
  Options() : use_keyframe(false) {}
  string path;
  string format;
  bool use_keyframe;
};

int TestTwoFrame();
int TestFramesInFolder(Options option);

int main(int argc, char** argv) {
  if (argc == 1)
    return TestTwoFrame();

  Options option;
  for (int i = 0; i < argc; ++i) {
    if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--path")) {
      option.path = argv[++i];
    } else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--format")) {
      option.format = argv[++i];
    } else if (!strcmp(argv[i], "--keyframe")) {
      option.use_keyframe = true;
      cout << "Using keyframe.\n";
    }
  }

  if (option.format.size() && option.path.size())
    return TestFramesInFolder(option);

  cout << "Error. Unknown arguments.\n";
  cout << "Usage: \n";
  cout << "       test\n";
  cout << "            -p, --path full_path \n";
  cout << "            -f, --format image format, e.g png, jpg\n";

  return -1;
}

int TestFramesInFolder(Options option) {
#ifndef __linux__
  cout << "Error: Test folder Not supported. Currently only support Linux.\n"
  return -1;
#endif
  vector<string> images;
  if (!GetImageNamesInFolder(option.path, option.format, images))
    return -1;

  if (images.size() < 2) {
    cout << "Error: Find only " << images.size() << " images.\n";
    return -1;
  }

  cout << "Testing with " << images.size() << " images.\n";

  cv::Mat image0 = cv::imread(images[0]);
  vector<cv::KeyPoint> kp0;
  cv::Mat desc0;

  if (!image0.data) {
    cerr << "Error: Unable to load image " << images[0] << endl;
    return -1;
  }

  cv::Ptr<cv::Feature2D> detector = cv::ORB::create(10000);
  vio::FeatureTracker *feature_tracker = vio::FeatureTracker::CreateFeatureTracker(detector);

  std::unique_ptr<vio::Frame> last_frame(new vio::Frame(image0));
  feature_tracker->TrackFirstFrame(*last_frame);

  KeyframeSelector keyframe_selector;

  cv::namedWindow( "result", cv::WINDOW_AUTOSIZE );

  for (int i = 1; i < images.size(); ++i) {
    cv::Mat image1 = cv::imread(images[i]);
    if (!image1.data) {
      cerr << "Error: Unable to load image " << images[0] << endl;
      return -1;
    }
    std::unique_ptr<vio::Frame> new_frame(new vio::Frame(image1));
    std::vector<cv::DMatch> matches;
    feature_tracker->TrackFrame(*last_frame, *new_frame, matches);

    std::cout << "Found " << matches.size() << " matches.\n"; 

    cv::Mat output_img = new_frame->GetImage().clone();

    int thickness = 2;
    for (int i = 0; i < matches.size(); ++i) {
      line(output_img, new_frame->GetFeatures().keypoints[matches[i].trainIdx].pt,
           last_frame->GetFeatures().keypoints[matches[i].queryIdx].pt,
           cv::Scalar(255, 0, 0), thickness);
    }

    cv::imshow("result", output_img); 
    cv::waitKey(0);

    if (option.use_keyframe) {
      if (keyframe_selector.isKeyframe(matches))
        last_frame = std::move(new_frame);
    }
  }

  return 0;  
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
