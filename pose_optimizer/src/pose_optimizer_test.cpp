#include "pose_optimizer.hpp"

// TODO: make the directory better
#include "../../feature_tracker/include/feature_tracker.hpp"
#include "../../landmark_server/include/landmark_server.hpp"

#include <opencv2/viz.hpp> 
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/sfm.hpp>

#include <string>

using namespace std;
using namespace cv;
using namespace cv::sfm;

int TestTwoViews(const char *left_image, const char *right_image);
int TestTwoViews() {
  string test_image_path =
      "/home/fan/Project/shumin_slam/pose_optimizer/test_data/resized_IMG_";
  string image0_name = test_image_path + "2889.jpg";
  string image1_name = test_image_path + "2890.jpg";
  return TestTwoViews(image0_name.c_str(), image1_name.c_str());
}
int TestThreeViews();

int main(int argc, char **argv) {
  if (argc == 1) return TestThreeViews();

  if (argc == 2 && !strcmp(argv[1], "--two")) return TestTwoViews();

  if (!strcmp(argv[1], "--two") && !strcmp(argv[2], "-l") &&
      !strcmp(argv[4], "-r"))
    return TestTwoViews(argv[3], argv[5]);

  cout << "Error. Unknown arguments.\n";
  cout << "Usage: \n";
  cout << "       test   (Test three views)\n";
  cout << "       test --two (Test two views)\n";
  cout << "       test --two -l path_to_left image -r path_to_right_image "
          "(Test Two Views)\n";


  Ptr<Feature2D> edetector = ORB::create(10000);
  Ptr<Feature2D> edescriber = xfeatures2d::DAISY::create();
  //Ptr<Feature2D> edescriber = xfeatures2d::LATCH::create(64, true, 4);

  return 0;
}

int TestTwoViews(const char *left_image, const char *right_image) {
  cv::Mat image0_origin = cv::imread(left_image);
  cv::Mat image1_origin = cv::imread(right_image);
  cv::Mat image0, image1;
  cv::cvtColor(image0_origin, image0, CV_BGR2GRAY);
  cv::cvtColor(image1_origin, image1, CV_BGR2GRAY);

  FeatureTracker feature_tracker;

  PoseOptimizer pose_optimizer(true);

  vector<vector<cv::KeyPoint> > kp(2);
  vector<cv::Mat> desc(2);
  vector<cv::DMatch> match;

  feature_tracker.DetectFeatureInFirstFrame(image0, kp[0], desc[0]);
  feature_tracker.TrackFeature(kp[0], desc[0], image1, kp[1], desc[1], match);

  cv::Mat img0to1;
  drawMatches(image0, kp[0], image1, kp[1], match, img0to1,
              cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));

  cv::namedWindow("result 0 to 1", cv::WINDOW_AUTOSIZE);
  cv::imshow("result 0 to 1", img0to1);
  cv::waitKey(0);

  vector<vector<cv::Vec2d> > feature_vectors(2);
  cv::Matx33d K_initial;
  vector<cv::Point3f> points3d;
  cv::Mat R_ests, t_ests;

  K_initial = cv::Matx33d(1914, 0, 640, 0, 1914, 360, 0, 0, 1);

  // Construct matching keypoints for initializer
  vector<cv::KeyPoint> kp0, kp1;
  for (int i = 0; i < match.size(); ++i) {
    kp0.push_back(kp[0][match[i].queryIdx]);
    kp1.push_back(kp[1][match[i].trainIdx]);
  }

  pose_optimizer.initialize3DPointsFromTwoViews(kp0, kp1, K_initial, points3d,
                                                R_ests, t_ests);

  return 0;
}

int TestThreeViews() {
  string test_image_path =
      "/home/fan/Project/shumin_slam/pose_optimizer/test_data/resized_IMG_";

  string image0_name = test_image_path + "2889.jpg";
  string image1_name = test_image_path + "2890.jpg";
  string image2_name = test_image_path + "2891.jpg";

  cv::Mat image0_origin = cv::imread(image0_name);
  cv::Mat image1_origin = cv::imread(image1_name);
  cv::Mat image2_origin = cv::imread(image2_name);

  if (!image0_origin.data || !image1_origin.data || !image2_origin.data) {
    std::cerr << "Unable to load image.\n";
    return -1;
  }

  cv::Mat image0, image1, image2;
  cv::cvtColor(image0_origin, image0, CV_BGR2GRAY);
  cv::cvtColor(image1_origin, image1, CV_BGR2GRAY);
  cv::cvtColor(image2_origin, image2, CV_BGR2GRAY);

  FeatureTracker feature_tracker;

  bool is_projective = true;
  PoseOptimizer pose_optimizer(is_projective);

  vector<vector<cv::KeyPoint> > kp(3);
  vector<cv::Mat> desc(3);
  // matches[i] is the matches of i --> i + 1
  vector<vector<cv::DMatch> > matches(2);

  feature_tracker.DetectFeatureInFirstFrame(image0, kp[0], desc[0]);
  feature_tracker.TrackFeature(kp[0], desc[0], image1, kp[1], desc[1],
                               matches[0]);
  feature_tracker.TrackFeature(kp[1], desc[1], image2, kp[2], desc[2],
                               matches[1]);

  cv::Mat img0to1, img1to2;
  drawMatches(image0, kp[0], image1, kp[1], matches[0], img0to1,
              cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
  drawMatches(image1, kp[1], image2, kp[2], matches[1], img1to2,
              cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));

  cv::namedWindow("result 0 to 1", cv::WINDOW_AUTOSIZE);
  cv::imshow("result 0 to 1", img0to1);
  cv::namedWindow("result 1 to 2", cv::WINDOW_AUTOSIZE);
  cv::imshow("result 1 to 2", img1to2);

  cv::waitKey(0);

  LandmarkServer landmark_server;
  landmark_server.AddFirstFrameFeature(kp[0]);
  landmark_server.AddNewFeatureAssociationToLastFrame(kp[1], matches[0]);
  landmark_server.AddNewFeatureAssociationToLastFrame(kp[2], matches[1]);

  landmark_server.PrintStats();

  vector<vector<cv::Vec2d> > feature_vectors(2);
  cv::Matx33d K_initial, K_final;
  vector<cv::Mat> points3d_estimated;
  vector<cv::Mat> Rs_est, ts_est;

  K_initial = cv::Matx33d(350, 0, 240, 0, 350, 360, 0, 0, 1);

  // Change data format of matched feature for initialization of 3d points.
  landmark_server.MakeFeatureVectorsForReconstruct(feature_vectors);

  pose_optimizer.initialize3DPointsFromViews(3, feature_vectors, K_initial,
                                             points3d_estimated, Rs_est, ts_est, K_final);

  /// Create 3D windows

  viz::Viz3d window("Coordinate Frame");
             window.setWindowSize(Size(500,500));
             window.setWindowPosition(Point(150,150));
             window.setBackgroundColor(); // black by default

  // Create the pointcloud
  cout << "Recovering points  ... ";

  // recover estimated points3d
  vector<Vec3f> point_cloud_est;
  for (int i = 0; i < points3d_estimated.size(); ++i)
    point_cloud_est.push_back(Vec3f(points3d_estimated[i]));

  cout << "[DONE]" << endl;


  /// Recovering cameras
  cout << "Recovering cameras ... ";

  vector<Affine3d> path;
  for (size_t i = 0; i < Rs_est.size(); ++i)
    path.push_back(Affine3d(Rs_est[i],ts_est[i]));

  cout << "[DONE]" << endl;


  /// Add the pointcloud
  if ( point_cloud_est.size() > 0 )
  {
    cout << "Rendering points   ... ";

    viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
    window.showWidget("point_cloud", cloud_widget);

    cout << "[DONE]" << endl;
  }
  else
  {
    cout << "Cannot render points: Empty pointcloud" << endl;
  }


  /// Add cameras
  if ( path.size() > 0 )
  {
    cout << "Rendering Cameras  ... ";

    window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
    window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K_final, 0.1, viz::Color::yellow()));

    window.setViewerPose(path[0]);

    cout << "[DONE]" << endl;
  }
  else
  {
    cout << "Cannot render the cameras: Empty path" << endl;
  }

  /// Wait for key 'q' to close the window
  cout << endl << "Press 'q' to close each windows ... " << endl;

  window.spin();


  return 0;
}
