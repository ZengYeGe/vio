#include "feature_tracker.hpp"
#include "keyframe_selector.hpp"

#ifdef __linux__
#include <dirent.h>
#endif

#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

int TestTwoFrame();

bool GetImageNamesInFolder(const char* path, const char* format, vector<string> &images);

int TestFramesInFolder(const char* path, const char* format);

int main(int argc, char** argv) {
  if (argc == 1)
    return TestTwoFrame();

  if (argc == 5 && (!strcmp(argv[1], "-p") || !strcmp(argv[1], "--path")) &&
                   (!strcmp(argv[3], "-f") || !strcmp(argv[3], "--format")))
    return TestFramesInFolder(argv[2], argv[4]);

  cout << "Error. Unknown arguments.\n";
  cout << "Usage: \n";
  cout << "       test\n";
  cout << "            -p, --path full_path \n";
  cout << "            -f, --format image format, e.g png, jpg\n";

  return 0;
}

int TestFramesInFolder(const char* path, const char* format) {
#ifndef __linux__
  cout << "Error: Test folder Not supported. Currently only support Linux.\n"
  return -1;
#endif
  vector<string> images;
  if (!GetImageNamesInFolder(path, format, images))
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

  FeatureTracker feature_tracker;
  // TODO: Add options to disable keyframe.
  KeyframeSelector keyframe_selector;

  feature_tracker.DetectFeatureInFirstFrame(image0, kp0, desc0);
  cv::namedWindow( "result", cv::WINDOW_AUTOSIZE );

  for (int i = 1; i < images.size(); ++i) {
    cv::Mat image1 = cv::imread(images[i]);
    if (!image1.data) {
      cerr << "Error: Unable to load image " << images[0] << endl;
      return -1;
    }
    vector<cv::KeyPoint> kp1;
    cv::Mat desc1;
    vector<cv::DMatch> matches;

    feature_tracker.TrackFeature(kp0, desc0, image1, kp1, desc1, matches);
    std::cout << "Found " << matches.size() << " matches.\n"; 

    cv::Mat output_img;
    // Draw two images ------------------
    // drawMatches(image0, kp0, image1, kp1, matches, output_img,
    //             cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
    // ----------------------------------------

    // Draw one image
    vector<cv::KeyPoint> kp_draw;
    for (int i = 0; i < matches.size(); ++i)
      kp_draw.push_back(kp1[matches[i].trainIdx]);

    // Only draw matched keypoints
    drawKeypoints(image1, kp_draw, output_img, cv::Scalar(255, 0, 0));

    int thickness = 3;
    for (int i = 0; i < matches.size(); ++i) {
      line(output_img, kp1[matches[i].trainIdx].pt,
           kp0[matches[i].queryIdx].pt, cv::Scalar(255, 0, 0), thickness);
    }

    cv::imshow("result", output_img); 
    cv::waitKey(0);

    if (keyframe_selector.isKeyframe(matches)) {
      kp0 = std::move(kp1);
      desc0 = desc1;
      image0 = image1;
    }
  }
  return 0;  
}

int TestTwoFrame() {
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
  feature_tracker.TrackFeature(kp0, desc0, image1, kp1, desc1, matches);

  std::cout << "Found " << matches.size() << " matches.\n"; 

  cv::Mat output_img;
  drawMatches(image0, kp0, image1, kp1, matches, output_img,
              cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
  
  cv::namedWindow( "result", cv::WINDOW_AUTOSIZE );
  cv::imshow("result", output_img); 
  cv::waitKey(0);

  return 0;
}

bool GetImageNamesInFolder(const char* path, const char* format, vector<string> &images) {
  struct dirent **file_list;
  int n = scandir(path, &file_list, 0, alphasort);
  if (n < 0) {
    cout << "Error: Unable to find directory " << path << endl;
    return false;
  } else {
    int format_len = strlen(format);
    string dir_path(path);
    for (int i = 0; i < n; ++i) {
      string file_name(file_list[i]->d_name);
      if (file_name.size() > format_len &&
          !file_name.compare(file_name.size() - format_len,
                             format_len, format)) {
        images.push_back(dir_path + '/' + file_name);
      }
    }
  }

  free(file_list);
  return true;
}


