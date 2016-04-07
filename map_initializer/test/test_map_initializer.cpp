#include "map_initializer.hpp"

#include <fstream>
#include <memory>
#include <string>

#include <opencv2/viz.hpp>
#include <opencv2/xfeatures2d.hpp>

// TODO: make the directory better
#include "../../feature_tracker/include/feature_tracker.hpp"
#include "../../feature_tracker/include/keyframe_selector.hpp"
#include "../../landmark_server/include/landmark_server.hpp"
#include "../../util/include/util.hpp"

using namespace std;
using namespace cv;

struct Options {
  Options() : use_keyframe(false) {}
  string path;
  string format;
  string match_file_name;
  bool use_keyframe;
};

// TODO: Add two views test.
int TestTwoFrameWithAccurateMatchFile(Options option);
int TestFramesInFolder(Options option);
void RunInitializer(vector<vector<cv::Vec2d> > &feature_vector);
// TODO: Put into util.
void VisualizeCamerasAndPoints(const cv::Matx33d &K,
                               const std::vector<cv::Mat> &Rs,
                               const std::vector<cv::Mat> &ts,
                               const std::vector<cv::Point3f> &points);

int main(int argc, char **argv) {
  Options option;
  for (int i = 0; i < argc; ++i) {
    if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--path")) {
      option.path = argv[++i];
    } else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--format")) {
      option.format = argv[++i];
    } else if (!strcmp(argv[i], "--keyframe")) {
      option.use_keyframe = true;
      cout << "Using keyframe.\n";
    } else if (!strcmp(argv[i], "--accuratetwo")) {
      option.match_file_name = argv[++i];
    }
  }

  if (option.match_file_name.size())
    return TestTwoFrameWithAccurateMatchFile(option);

  if (option.format.size() && option.path.size())
    return TestFramesInFolder(option);

  cout << "Error. Unknown arguments.\n";
  cout << "Usage: \n";
  cout << "       test\n";
  cout << "            -p, --path full_path \n";
  cout << "            -f, --format image format, e.g png, jpg\n";
  cout << "            --keyframe, select key frame\n";
  // TODO: Add test two frames with good baseline and bad baseline.
  cout << "            --accuratetwo  path_to_match_file, test using accurate "
          "matches.\n";

  return -1;
}
int TestTwoFrameWithAccurateMatchFile(Options options) {
  cv::Mat_<double> x1, x2;
  int npts;
  ifstream myfile(options.match_file_name.c_str());
  if (!myfile.is_open()) {
    cout << "Unable to read file: " << options.match_file_name << endl;
    exit(0);
  }
  vector<vector<cv::Vec2d> > feature_vectors(2);

  string line;

  // Read number of points
  getline(myfile, line);
  npts = (int)atof(line.c_str());

  feature_vectors[0].resize(npts);
  feature_vectors[1].resize(npts);

  x1 = Mat_<double>(2, npts);
  x2 = Mat_<double>(2, npts);

  // Read the point coordinates
  for (int i = 0; i < npts; ++i) {
    getline(myfile, line);
    stringstream s(line);
    string cord;

    s >> cord;
    x1(0, i) = atof(cord.c_str());
    s >> cord;
    x1(1, i) = atof(cord.c_str());

    s >> cord;
    x2(0, i) = atof(cord.c_str());
    s >> cord;
    x2(1, i) = atof(cord.c_str());

    feature_vectors[0][i][0] = x1(0, i);
    feature_vectors[0][i][1] = x1(1, i);
    feature_vectors[1][i][0] = x2(0, i);
    feature_vectors[1][i][1] = x2(1, i);

  }
  myfile.close();

  RunInitializer(feature_vectors);

  return 0;
}
int TestFramesInFolder(Options option) {
#ifndef __linux__
  cout << "Error: Test folder Not supported. Currently only support "
          "Linux.\n" return -1;
#endif
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

  // TODO: Add option for selecting feature detector.
  cv::Ptr<cv::Feature2D> detector = cv::ORB::create(10000);
  cv::Ptr<cv::Feature2D> descriptor = cv::xfeatures2d::DAISY::create();
  // vio::FeatureTracker *feature_tracker =
  // vio::FeatureTracker::CreateFeatureTracker(detector);
  vio::FeatureTracker *feature_tracker =
      vio::FeatureTracker::CreateFeatureTracker(detector, descriptor);

  std::unique_ptr<vio::Frame> last_frame(new vio::Frame(image0));
  feature_tracker->TrackFirstFrame(*last_frame);

  KeyframeSelector keyframe_selector;

  LandmarkServer landmark_server;
  landmark_server.AddFirstFrameFeature(last_frame->GetFeatures().keypoints);

  cv::namedWindow("tracking_result", cv::WINDOW_AUTOSIZE);
  int num_frames = 0;
  for (int i = 1; i < images.size(); ++i) {
    cv::Mat image1 = cv::imread(images[i]);
    if (!image1.data) {
      cerr << "Error: Unable to load image " << images[0] << endl;
      return -1;
    }
    std::unique_ptr<vio::Frame> new_frame(new vio::Frame(image1));
    std::vector<cv::DMatch> matches;
    feature_tracker->TrackFrame(*last_frame, *new_frame, matches);

    std::cout << "Found " << matches.size() << " matches.\n";

    cv::Mat output_img = new_frame->GetImage().clone();

    int thickness = 2;
    for (int i = 0; i < matches.size(); ++i) {
      line(output_img,
           new_frame->GetFeatures().keypoints[matches[i].trainIdx].pt,
           last_frame->GetFeatures().keypoints[matches[i].queryIdx].pt,
           cv::Scalar(255, 0, 0), thickness);
    }

    cv::imshow("tracking_result", output_img);
    cv::waitKey(0);

    if (option.use_keyframe) {
      if (!keyframe_selector.isKeyframe(matches)) continue;
    }

    // Add to landmark server
    landmark_server.AddNewFeatureAssociationToLastFrame(
        new_frame->GetFeatures().keypoints, matches);
    last_frame = std::move(new_frame);
    num_frames++;
  }

  landmark_server.PrintStats();

  // TODO: Replace with RunInitializer

  vector<vector<cv::Vec2d> > feature_vectors(images.size());
  // TODO: Add namespace vio to landmakr_server
  // TODO: Verify this is correct
  landmark_server.MakeFeatureVectorsForReconstruct(feature_vectors);

  RunInitializer(feature_vectors);

  return 0;
}

void RunInitializer(vector<vector<cv::Vec2d> > &feature_vectors) {
  cv::Matx33d K_initial;
  vector<cv::Point3f> points3d;
  vector<cv::Mat> Rs_est, ts_est;
  K_initial = cv::Matx33d(350, 0, 240, 0, 350, 360, 0, 0, 1);

  // TODO: Add option to select initializer.
  //vio::MapInitializer *map_initializer =
  //    vio::MapInitializer::CreateMapInitializer(vio::LIVMV);
  vio::MapInitializer *map_initializer =
      vio::MapInitializer::CreateMapInitializer(vio::NORMALIZED8POINTFUNDAMENTAL);
  map_initializer->Initialize(feature_vectors, cv::Mat(K_initial), points3d,
                              Rs_est, ts_est);

  VisualizeCamerasAndPoints(K_initial, Rs_est, ts_est, points3d);
}

void VisualizeCamerasAndPoints(const cv::Matx33d &K,
                               const std::vector<cv::Mat> &Rs,
                               const std::vector<cv::Mat> &ts,
                               const std::vector<cv::Point3f> &points) {
  /// Create 3D windows

  viz::Viz3d window("Coordinate Frame");
  window.setWindowSize(Size(500, 500));
  window.setWindowPosition(Point(150, 150));
  window.setBackgroundColor();  // black by default

  // Create the pointcloud
  cout << "Recovering points  ... ";

  // recover estimated points3d
  vector<Vec3f> point_cloud_est;
  for (int i = 0; i < points.size(); ++i)
    point_cloud_est.push_back(Vec3f(points[i]));

  cout << "[DONE]" << endl;

  /// Recovering cameras
  cout << "Recovering cameras ... ";

  vector<Affine3d> path;
  for (size_t i = 0; i < Rs.size(); ++i) path.push_back(Affine3d(Rs[i], ts[i]));

  cout << "[DONE]" << endl;

  /// Add the pointcloud
  if (point_cloud_est.size() > 0) {
    cout << "Rendering points   ... ";

    viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
    window.showWidget("point_cloud", cloud_widget);

    cout << "[DONE]" << endl;
  } else {
    cout << "Cannot render points: Empty pointcloud" << endl;
  }

  /// Add cameras
  if (path.size() > 0) {
    cout << "Rendering Cameras  ... ";

    window.showWidget("cameras_frames_and_lines",
                      viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1,
                                       viz::Color::green()));
    window.showWidget(
        "cameras_frustums",
        viz::WTrajectoryFrustums(path, K, 0.1, viz::Color::yellow()));

    window.setViewerPose(path[0]);

    cout << "[DONE]" << endl;
  } else {
    cout << "Cannot render the cameras: Empty path" << endl;
  }

  /// Wait for key 'q' to close the window
  cout << endl
       << "Press 'q' to close each windows ... " << endl;

  window.spin();
}
