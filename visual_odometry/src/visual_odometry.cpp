#include "visual_odometry.hpp"

namespace vio {

VisualOdometry::VisualOdometry(const VisualOdometryConfig &config) 
    : status_(UNINITED) {

  camera_model_ = new CameraModel(config.camera_model_params);

  feature_tracker_ = FeatureTracker::CreateFeatureTracker(config.feature_tracker_option);
  map_initializer_ = MapInitializer::CreateMapInitializer(config.map_initializer_option);

  pnp_estimator_ = PnPEstimator::CreatePnPEstimator(ITERATIVE);
}

bool VisualOdometry::IsInited() {
  if (!feature_tracker_ || !map_initializer_ || !map_initializer_)
    return false;
  return true;
}

bool VisualOdometry::AddNewRawImage(const cv::Mat &img) {
  return true;
}

} // vio
