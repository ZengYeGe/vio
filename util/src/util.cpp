#include "util.hpp"

bool GetImageNamesInFolder(const std::string &path, const std::string &format,
                           std::vector<std::string> &images) {
  struct dirent **file_list;
  int n = scandir(path.c_str(), &file_list, 0, alphasort);
  if (n < 0) {
    std::cerr << "Error: Unable to find directory " << path << std::endl;
    return false;
  } else {
    for (int i = 0; i < n; ++i) {
      std::string file_name(file_list[i]->d_name);
      if (file_name.size() > format.size() &&
          !file_name.compare(file_name.size() - format.size(), format.size(),
                             format)) {
        images.push_back(path + '/' + file_name);
      }
    }
  }

  free(file_list);
  return true;
}

void VisualizeCamerasAndPoints(const cv::Matx33d &K,
                               const std::vector<cv::Mat> &Rs,
                               const std::vector<cv::Mat> &ts,
                               const std::vector<cv::Point3f> &points) {
  /// Create 3D windows
  cv::viz::Viz3d window("Coordinate Frame");
  window.setWindowSize(cv::Size(500, 500));
  window.setWindowPosition(cv::Point(150, 150));
  window.setBackgroundColor();  // black by default

  // Create the pointcloud
  std::cout << "Recovering points  ... ";

  // recover estimated points3d
  std::vector<cv::Vec3f> point_cloud_est;
  for (int i = 0; i < points.size(); ++i)
    point_cloud_est.push_back(cv::Vec3f(points[i]));

  std::cout << "[DONE]" << std::endl;

  /// Recovering cameras
  std::cout << "Recovering cameras ... ";

  std::vector<cv::Affine3d> path;
  for (size_t i = 0; i < Rs.size(); ++i)
    path.push_back(cv::Affine3d(Rs[i], ts[i]));

  std::cout << "[DONE]" << std::endl;

  /// Add the pointcloud
  if (point_cloud_est.size() > 0) {
    std::cout << "Rendering points   ... ";

    cv::viz::WCloud cloud_widget(point_cloud_est, cv::viz::Color::green());
    window.showWidget("point_cloud", cloud_widget);

    std::cout << "[DONE]" << std::endl;
  } else {
    std::cout << "Cannot render points: Empty pointcloud" << std::endl;
  }

  /// Add cameras
  if (path.size() > 0) {
    std::cout << "Rendering Cameras  ... ";

    window.showWidget("cameras_frames_and_lines",
                      cv::viz::WTrajectory(path, cv::viz::WTrajectory::BOTH,
                                           0.1, cv::viz::Color::green()));
    window.showWidget(
        "cameras_frustums",
        cv::viz::WTrajectoryFrustums(path, K, 0.1, cv::viz::Color::yellow()));

    window.setViewerPose(path[0]);

    std::cout << "[DONE]" << std::endl;
  } else {
    std::cout << "Cannot render the cameras: Empty path" << std::endl;
  }

  /// Wait for key 'q' to close the window
  std::cout << std::endl << "Press 'q' to close each windows ... " << std::endl;

  window.spin();
}
