#include <fstream>
#include <memory>
#include <string>

#include <opencv2/xfeatures2d.hpp>

#include "visual_odometry.hpp"

using namespace std;
using namespace cv;

struct Options {
  Options() : use_keyframe(false) {}
  string path;
  string format;
  string match_file_name;
  bool use_keyframe;
  string config_filename;
  string calibration_filename;
};

// int TestTwoFrameWithAccurateMatchFile(Options option);

int TestFramesInFolder(const Options &option, vio::VisualOdometryConfig &vo_config);

int TestVideo(const Options &option, const vio::VisualOdometryConfig &config);

void RunInitializer(vector<vector<cv::Vec2d> > &feature_vector);
