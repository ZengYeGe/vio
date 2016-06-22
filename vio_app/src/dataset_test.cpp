#include "vio_app.hpp"

int TestFramesInFolder(const Options &option,
                       vio::VisualOdometryConfig &vo_config) {
#ifndef __linux__
  cerr << "Error: Test folder Not supported. Currently only support "
          "Linux.\n" return -1;
#endif

  const std::string calibration_file_name =
      option.path + "/dataset_config.yaml";
  if (!vo_config.SetUpCameraFromFile(calibration_file_name)) {
    cerr << "Error: Couldn't find dataset config file in the folder.\n";
    return -1;
  }

  vio::VisualOdometry vo(vo_config);
  if (!vo.IsInited()) return -1;

  // -------------- Load images
  vector<string> images;
  if (!GetImageNamesInFolder(option.path, option.format, images)) return -1;

  if (images.size() < 2) {
    cout << "Error: Find only " << images.size() << " images.\n";
    return -1;
  }
  cout << "Testing with " << images.size() << " images.\n";

  cv::namedWindow("tracking_result", cv::WINDOW_AUTOSIZE);

  for (int i = 0; i < images.size(); ++i) {
    cv::Mat image = cv::imread(images[i]);
    if (!image.data) {
      cerr << "Error: Unable to load image " << images[0] << endl;
      return -1;
    }

    cout << "Adding image " << i << std::endl;

    vio::FramePose current_pose;
    if (vio::ERROR == vo.TrackNewRawImage(image, current_pose)) return -1;
  }

  vo.VisualizeMap();
  /*
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

    VisualizeCamerasAndPoints(K_initial, R_all, t_all, points3d_all);
  */

  /*
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
  */
  return 0;
}
