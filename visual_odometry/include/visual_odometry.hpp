#ifndef VISUAL_ODOMETRY_
#define VISUAL_ODOMETRY_

#include <opencv2/opencv.hpp>

#include "camera_model.hpp"
#include "feature_tracker.hpp"
#include "graph_optimizer.hpp"
#include "keyframe_selector.hpp"
#include "mapdata.hpp"
#include "map_initializer.hpp"
#include "multiview.hpp"
#include "pnp_estimator.hpp"
#include "util.hpp"

namespace vio {

struct VisualOdometryConfig {
 public:
  bool SetUpFromFile(const std::string &config_file) {
    cv::FileStorage pipeline_config;
    pipeline_config.open(config_file, cv::FileStorage::READ);
    if (!pipeline_config.isOpened()) {
      cerr << "Error: Couldn't open pipeline config file.\n";
      return false;
    }
    pipeline_config["FeatureTracker"] >> feature_tracker_option;
    pipeline_config["FeatureMatcher"] >> feature_matcher_option;
    pipeline_config["MapInitializer"] >> map_initializer_option;

    // Setup pipeline
    const cv::FileNode &node = pipeline_config["VisualOdometry"];
    viz_tracking = (int)node["VisualizeTracking"] == 1;
    viz_time_per_frame = (int)node["TimePerFrame"];
    viz_landmarks = (int)node["VisualizeLandmarks"] == 1;
    viz_landmarks_every_frame = (int)node["VisualizeLandmarksEveryNumFrame"];
    optimize_every_frame = (int)node["OptimizeEveryNumFrame"];
    return true;
  }

  bool SetUpCameraFromFile(const std::string &camera_config_file) {
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
  FeatureMatcherOptions feature_matcher_option;
  MapInitializerOptions map_initializer_option;

  int optimize_every_frame;

  bool viz_tracking;
  int viz_time_per_frame; 
  bool viz_landmarks;
  int viz_landmarks_every_frame;
};

enum VO_Status {
  UNINITED = 0,
  INITED,
};

enum TrackingStatus {
  ERROR = 0,
  TRACKING_AVAILABLE = 1, 
  TRACKING_NOT_AVAILABLE = 2
};

class VisualOdometry {
 public:
  VisualOdometry(const VisualOdometryConfig &config);
  VisualOdometry() = delete;

  bool IsInited();

  // TODO: Avoid copy
  TrackingStatus TrackNewRawImage(const cv::Mat &img, FramePose &pose);

  void VisualizeMap();

 // Methods
 private:
  bool AddFirstFrame(std::unique_ptr<ImageFrame> frame);
  bool AddNewFrame(std::unique_ptr<ImageFrame> frame);

  bool InitializeLandmarks();
  bool EstimateLastFrame();

  bool OptimizeMap();

  void PlotTracking(const ImageFrame &frame0, const ImageFrame &frame1,
                    const std::vector<cv::DMatch> &matches);
 // Variables
 private:

  VO_Status status_;

  const CameraModel *camera_model_;

  FeatureTracker *feature_tracker_;

  MapInitializer *map_initializer_;

  PnPEstimator *pnp_estimator_;

  GraphOptimizer *optimizer_;

  KeyframeSelector keyframe_selector_;

  Mapdata map_;

  Keyframe *current_frame_;

  int optimize_every_frame_;

  bool plot_tracking_;
  int tracking_wait_time_;

  bool plot_3d_landmarks_;
  int plot_3d_landmarks_every_frame_;
};

} // vio

#endif // VISUAL_ODOMETRY_
