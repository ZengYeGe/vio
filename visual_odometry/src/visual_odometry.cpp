#include "visual_odometry.hpp"

#include <iostream>

namespace vio {

VisualOdometry::VisualOdometry(const VisualOdometryConfig &config) 
    : status_(UNINITED) {

  camera_model_ = new CameraModel(config.camera_model_params);

  feature_tracker_ = FeatureTracker::CreateFeatureTracker(config.feature_tracker_option);
  map_initializer_ = MapInitializer::CreateMapInitializer(config.map_initializer_option);

  pnp_estimator_ = PnPEstimator::CreatePnPEstimator(ITERATIVE);
  optimizer_ = GraphOptimizer::CreateGraphOptimizer(vio::CERES);

  if (IsInited())
    status_ = INITED;
}

bool VisualOdometry::IsInited() {
  if (!feature_tracker_) {
    std::cerr << "Error: Feature tracker not created.\n";
    return false;
  }
  if (!map_initializer_) {
    std::cerr << "Error: Map initializer not created.\n";
    return false;
  }
  if (!pnp_estimator_) {
    std::cerr << "Error: PnP estimator not created.\n";
    return false;
  }
  return true;
}

TrackingStatus VisualOdometry::TrackNewRawImage(const cv::Mat &img, FramePose &pose) {
  if (!IsInited()) return ERROR;

  std::unique_ptr<ImageFrame> frame(new ImageFrame(img));
  

  // If no image has been added.
  if (map_.state() == Mapdata::WAIT_FOR_FIRSTFRAME) {
    AddFirstFrame(std::move(frame));
    return TRACKING_NOT_AVAILABLE;
  }

  if (!AddNewFrame(std::move(frame))) return ERROR;

  // TODO: Move initialization out of mapdata, map should only save mapped data.
  // TODO: Put frames before initialization in a temporary vector.
  if (map_.state() == Mapdata::WAIT_FOR_INIT) {
      if (!InitializeLandmarks())
        return ERROR;
  }

  if (map_.state() == Mapdata::INITIALIZED) {
    pose = map_.GetLastKeyframe().pose();
  }

  return TRACKING_NOT_AVAILABLE; 
}

bool VisualOdometry::AddFirstFrame(std::unique_ptr<ImageFrame> frame) {
  feature_tracker_->TrackFirstFrame(*frame);
  // TODO: Doesn't make sense to do move a lot
  std::unique_ptr<Keyframe> first_keyframe(new Keyframe(std::move(frame)));
  if (!map_.AddFirstKeyframe(std::move(first_keyframe)))
    return false;
  return true;
}

bool VisualOdometry::AddNewFrame(std::unique_ptr<ImageFrame> frame) {
  std::vector<cv::DMatch> matches;
  if (!feature_tracker_->TrackFrame(map_.GetLastKeyframe().image_frame(),
                                     *frame, matches)) {
    std::cerr << "Error: Track new frame failed.\n";
    return false;
  }
/*
  // TODO: Refine Keyframe Selector.
  // TODO: Add select keyframe for initialization
  if (!keyframe_selector_.isKeyframe(matches)) {
    std::cout << "Skipped a frame. Not selected as keyframe.\n";
    return true; 
  }
*/
  std::unique_ptr<Keyframe> new_keyframe(new Keyframe(std::move(frame)));
  if (!map_.AddNewKeyframeMatchToLastKeyframe(std::move(new_keyframe), matches))
    return false;

  if (map_.state() != Mapdata::INITIALIZED)
    return true;

  // Estimate last frame
  EstimateLastFrame();
  map_.PrintStats();

  if (map_.num_frame() % 5 == 0)
    OptimizeMap();

  return true;
}

bool VisualOdometry::InitializeLandmarks() {
  vector<vector<cv::Vec2d> > feature_vectors;
  if (!map_.PrepareInitializationData(feature_vectors)) {
    std::cerr << "Error: Prepare data for initialization failed.\n";
    return false;
  }

  // Initialization
  vector<cv::Point3f> points_3d_est;
  // Validation flag for |points_3d_est|
  vector<bool> points_3d_mask;
  vector<cv::Mat> R_seq_est, t_seq_est;

  // TODO: Undistort using camera model then pass to do initialization.
  if (!map_initializer_->Initialize(feature_vectors, camera_model_->K(), points_3d_est,
                              points_3d_mask, R_seq_est, t_seq_est))
    return false;
  if (!map_.AddInitialization(points_3d_est, points_3d_mask, R_seq_est, t_seq_est))
    return false;

  VisualizeMap();

  return true;
}

bool VisualOdometry::EstimateLastFrame() {
  // -------------- Add landmarks
  std::vector<cv::Point3f> points3d;
  std::vector<cv::Point2f> points2d;
  std::vector<int> points_index;

  map_.PrepareEstimateLastFramePoseData(points3d, points2d, points_index);

  std::vector<bool> inliers;
  cv::Mat R;
  cv::Mat t;
  pnp_estimator_->EstimatePose(points2d, points3d, camera_model_->K(), inliers,
                               R, t);
  map_.SetLastFramePose(R, t);

  // Add new landmarks
  std::vector<cv::Vec2d> kp0, kp1;
  FramePose pose0, pose1;
  map_.PrepareUninitedPointsFromLastTwoFrames(kp0, kp1, pose0, pose1);

  std::vector<cv::Point3f> new_points3d;
  std::vector<bool> new_points3d_mask;
  TriangulatePoints(kp0, kp1, camera_model_->K(), pose0.R, pose0.t,
                    pose1.R, pose1.t, new_points3d, new_points3d_mask);

    // TODO: points not added
  if (!map_.AddInitedPoints(new_points3d, new_points3d_mask))
    return false;

  return true;
}

bool VisualOdometry::OptimizeMap() {
   // Optimization
  std::vector<cv::Mat> Rs;
  std::vector<cv::Mat> ts;
  std::vector<cv::Point3f> points;
  std::vector<int> obs_camera_idx;
  std::vector<int> obs_point_idx;
  std::vector<cv::Vec2d> obs_feature;
  map_.PrepareOptimization(Rs, ts, points, obs_camera_idx, obs_point_idx,
                              obs_feature);

  if (!optimizer_->Optimize(camera_model_->K(), Rs, ts, points, obs_camera_idx,
                      obs_point_idx, obs_feature))
    return false;

  if (!map_.ApplyOptimization(Rs, ts, points))
    return false;

  VisualizeMap();

  return true;
}

void VisualOdometry::VisualizeMap() {
  std::vector<cv::Mat> Rs, ts;
  std::vector<cv::Point3f> pts_3d;
  for (int i = 0; i < map_.num_frame(); ++i) {
    const Keyframe &frame = map_.keyframe(i);
    Rs.push_back(frame.GetRot());
    ts.push_back(frame.GetT());
  }
  for (int i = 0; i < map_.num_landmark(); ++i) {
    pts_3d.push_back(map_.landmark(i).position);
  }

  VisualizeCamerasAndPoints(camera_model_->K(), Rs, ts, pts_3d);
}

} // vio
