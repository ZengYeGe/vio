#include "vio_app.hpp"

int TestFramesInFolder(Options option) {
#ifndef __linux__
  cerr << "Error: Test folder Not supported. Currently only support "
          "Linux.\n" return -1;
#endif
  cv::FileStorage dataset_config;
  dataset_config.open(option.path + "/dataset_config.yaml", FileStorage::READ);
  if (!dataset_config.isOpened()) {
    cerr << "Error: Couldn't find dataset config file in the folder.\n";
    return -1;
  }

  cv::FileStorage pipeline_config;
  pipeline_config.open(option.config_filename, FileStorage::READ);
  if (!pipeline_config.isOpened()) {
    cerr << "Error: Couldn't find pipeline config file.\n";
    return -1;
  }

  // -------------- Load images
  vector<string> images;
  if (!GetImageNamesInFolder(option.path, option.format, images)) return -1;

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

  // --------------- Load parameters
  cv::Mat K_initial;
  dataset_config["K"] >> K_initial;
  cout << "Camera intrinsics: \n" << K_initial << std::endl;


  vio::FeatureTrackerOptions feature_tracker_option;
  pipeline_config["FeatureTracker"] >> feature_tracker_option;

  vio::MapInitializerOptions map_initializer_option;


  // ------------- Create modules 
  vio::FeatureTracker *feature_tracker =
      vio::FeatureTracker::CreateFeatureTracker(feature_tracker_option);

  KeyframeSelector keyframe_selector;

  vio::MapInitializer *map_initializer =
      vio::MapInitializer::CreateMapInitializer(map_initializer_option);

  vio::Map vio_map;

  vio::PnPEstimator *pnp_estimator =
      vio::PnPEstimator::CreatePnPEstimator(vio::ITERATIVE);
 

  // ------------ Start process
  if (!feature_tracker) {
    cerr << "Error: Failed to create feature tracker.\n";
    return -1;
  }

  std::unique_ptr<vio::ImageFrame> last_frame(new vio::ImageFrame(image0));

  feature_tracker->TrackFirstFrame(*last_frame);
  std::cout << "Found " << last_frame->keypoints().size() << " features.\n";

 // TODO: Doesn't make sense to do move a lot
  std::unique_ptr<vio::Keyframe> first_keyframe(
      new vio::Keyframe(std::move(last_frame)));
  vio_map.AddFirstKeyframe(std::move(first_keyframe));

  vector<cv::Mat> R_all;
  vector<cv::Mat> t_all;
  vector<cv::Point3f> points3d_all;

  cv::namedWindow("tracking_result", cv::WINDOW_AUTOSIZE);
  int num_frames = 1;
  for (int i = 1; i < images.size(); ++i) {
    // ---------------------- Load new frame --------------------------
    cv::Mat image1 = cv::imread(images[i]);
    if (!image1.data) {
      cerr << "Error: Unable to load image " << images[i] << endl;
      return -1;
    }
    std::unique_ptr<vio::ImageFrame> new_frame(new vio::ImageFrame(image1));

    // ----------------------Find match ------------------------------------
    std::vector<cv::DMatch> matches;
    if (!feature_tracker->TrackFrame(vio_map.GetLastKeyframe().image_frame(),
                                     *new_frame, matches))
      return -1;
    std::cout << "Found " << matches.size() << " matches.\n";

    // -------------------- Show tracking -------------------------------
    cv::Mat output_img = new_frame->GetImage().clone();
    int thickness = 2;
    for (int i = 0; i < matches.size(); ++i) {
      line(output_img, new_frame->keypoints()[matches[i].trainIdx].pt,
           vio_map.GetLastKeyframe()
               .image_frame()
               .keypoints()[matches[i].queryIdx]
               .pt,
           cv::Scalar(255, 0, 0), thickness);
    }
    cv::imshow("tracking_result", output_img);
    cv::waitKey(50);
    // ---------------------------------------------------------------

    if (option.use_keyframe) {
      if (!keyframe_selector.isKeyframe(matches)) continue;
    }

    // ----------------- Add to map ---------------------------------------
    std::unique_ptr<vio::Keyframe> new_keyframe(
        new vio::Keyframe(std::move(new_frame)));
    vio_map.AddNewKeyframeMatchToLastKeyframe(std::move(new_keyframe), matches);
    num_frames++;

    // -------------------- Initialization -------------------------------------
    if (num_frames == 2) {
      vector<vector<cv::Vec2d> > feature_vectors(images.size());
      vio_map.PrepareInitializationData(feature_vectors);

      // Initialization
      vector<cv::Point3f> points3d;
      vector<bool> points3d_mask;
      vector<cv::Mat> Rs_est, ts_est;
      // TODO: Add option to select initializer.
      //  vio::MapInitializer *map_initializer =
      //      vio::MapInitializer::CreateMapInitializer(vio::LIVMV);
      map_initializer->Initialize(feature_vectors, K_initial, points3d,
                                  points3d_mask, Rs_est, ts_est);

      vio_map.AddInitialization(points3d, points3d_mask, Rs_est, ts_est);

      // Add visualize to vio_map
      VisualizeCamerasAndPoints(K_initial, Rs_est, ts_est, points3d);

      R_all = std::move(Rs_est);
      t_all = std::move(ts_est);
      points3d_all = std::move(points3d);

      vio_map.PrintStats();

      continue;
    }

    // -------------- Add landmarks
    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;
    std::vector<int> points_index;

    vio_map.PrepareEstimateLastFramePoseData(points3d, points2d, points_index);

    std::vector<bool> inliers;
    cv::Mat R;
    cv::Mat t;
    pnp_estimator->EstimatePose(points2d, points3d, K_initial, inliers,
                                R, t);
    vio_map.SetLastFramePose(R, t);

    // Add new landmarks
    std::vector<cv::Vec2d> kp0, kp1;
    vio::FramePose pose0, pose1;
    vio_map.PrepareUninitedPointsFromLastTwoFrames(kp0, kp1, pose0, pose1);

    std::vector<cv::Point3f> new_points3d;
    std::vector<bool> new_points3d_mask;
    vio::TriangulatePoints(kp0, kp1, K_initial, pose0.R, pose0.t,
                           pose1.R, pose1.t, new_points3d, new_points3d_mask);

    // TODO: points not added
    vio_map.AddInitedPoints(new_points3d, new_points3d_mask);

    R_all.push_back(R);
    t_all.push_back(t);
    points3d_all.insert(points3d_all.end(), new_points3d.begin(),
                        new_points3d.end());
    vio_map.PrintStats();
  }

  VisualizeCamerasAndPoints(K_initial, R_all, t_all, points3d_all);

  // Optimization
  std::vector<cv::Mat> Rs;
  std::vector<cv::Mat> ts;
  std::vector<cv::Point3f> points;
  std::vector<int> obs_camera_idx;
  std::vector<int> obs_point_idx;
  std::vector<cv::Vec2d> obs_feature;
  vio_map.PrepareOptimization(Rs, ts, points, obs_camera_idx, obs_point_idx,
                              obs_feature);

  vio::GraphOptimizer *optimizer =
      vio::GraphOptimizer::CreateGraphOptimizer(vio::CERES);

  optimizer->Optimize(K_initial, Rs, ts, points, obs_camera_idx,
                      obs_point_idx, obs_feature);

  vio_map.ApplyOptimization(Rs, ts, points);

  VisualizeCamerasAndPoints(K_initial, Rs, ts, points);

  return 0;
}
