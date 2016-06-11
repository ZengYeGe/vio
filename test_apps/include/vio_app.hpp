#include <fstream>
#include <memory>
#include <string>

#include <opencv2/xfeatures2d.hpp>


#include "feature_tracker.hpp"
#include "keyframe_selector.hpp"
#include "map_initializer.hpp"
#include "pnp_estimator.hpp"
#include "graph_optimizer.hpp"
#include "util.hpp"
#include "multiview.hpp"
#include "map.hpp"

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

class PipelineConfig {
 public:
  bool SetUpFromFile(std::string config_file) {
    cv::FileStorage pipeline_config;
    pipeline_config.open(config_file, FileStorage::READ);
    if (!pipeline_config.isOpened()) {
      cerr << "Error: Couldn't find pipeline config file.\n";
      return false;
    }
    pipeline_config["FeatureTracker"] >> feature_tracker_option;
    pipeline_config["MapInitializer"] >> map_initializer_option;
    return true;
  }

  vio::FeatureTrackerOptions feature_tracker_option;
  vio::MapInitializerOptions map_initializer_option;

};

int TestTwoFrameWithAccurateMatchFile(Options option);

int TestFramesInFolder(const Options &option, const PipelineConfig &config);

int TestVideo(const Options &option, const PipelineConfig &config);

void RunInitializer(vector<vector<cv::Vec2d> > &feature_vector);
