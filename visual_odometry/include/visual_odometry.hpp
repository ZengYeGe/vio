#ifndef VISUAL_ODOMETRY_
#define VISUAL_ODOMETRY_

#include <opencv2/opencv.hpp>

#include "camera_model.hpp"
#include "feature_tracker.hpp"
#include "graph_optimizer.hpp"
#include "keyframe_selector.hpp"
#include "map.hpp"
#include "map_initializer.hpp"
#include "multiview.hpp"
#include "pnp_estimator.hpp"
#include "util.hpp"

namespace vio {

struct VisualOdometryConfig {
 public:
  bool SetUpFromFile(std::string config_file) {
    cv::FileStorage pipeline_config;
    pipeline_config.open(config_file, cv::FileStorage::READ);
    if (!pipeline_config.isOpened()) {
      cerr << "Error: Couldn't open pipeline config file.\n";
      return false;
    }
    pipeline_config["FeatureTracker"] >> feature_tracker_option;
    pipeline_config["MapInitializer"] >> map_initializer_option;
    return true;
  }

  bool SetUpCamera(std::string camera_config_file) {
    cv::FileStorage camera_config;
    camera_config.open(camera_config_file, cv::FileStorage::READ);
    if (!camera_config.isOpened()) {
      cerr << "Error: Couldn't open camera config file.\n";
      return false;
    }
    camera_config["CameraModel"] >> camera_model_params;
    return true;
  }

  CameraModelParams camera_model_params;

  FeatureTrackerOptions feature_tracker_option;
  MapInitializerOptions map_initializer_option;

};

enum VO_Status {
  UNINITED = 0,
  INITED,
  TRACKING
};

class VisualOdometry {
 public:
  VisualOdometry(const VisualOdometryConfig &config);
  VisualOdometry() = delete;

  bool IsInited();

  // TODO: Avoid copy
  bool AddNewRawImage(const cv::Mat &img);

 // Methods
 private:
  

 // Variables
 private:

  VO_Status status_;

  const CameraModel *camera_model_;

  const FeatureTracker *feature_tracker_;

  const MapInitializer *map_initializer_;

  const PnPEstimator *pnp_estimator_;

  KeyframeSelector keyframe_selector_;

  Map map_;

};

} // vio

#endif // VISUAL_ODOMETRY_
