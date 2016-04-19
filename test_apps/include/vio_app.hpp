#include <fstream>
#include <memory>
#include <string>

#include <opencv2/xfeatures2d.hpp>

// TODO: make the directory better
#include "../../feature_tracker/include/feature_tracker.hpp"
#include "../../feature_tracker/include/keyframe_selector.hpp"
#include "../../landmark_server/include/landmark_server.hpp"
#include "../../map_initializer/include/map_initializer.hpp"
#include "../../util/include/util.hpp"
#include "../../mapdata/include/map.hpp"

using namespace std;
using namespace cv;

struct Options {
  Options() : use_keyframe(false) {}
  string path;
  string format;
  string match_file_name;
  bool use_keyframe;
};

int TestTwoFrameWithAccurateMatchFile(Options option);

int TestFramesInFolder(Options option);

void RunInitializer(vector<vector<cv::Vec2d> > &feature_vector);

