#include "pose_optimizer.hpp"

// TODO: make the directory better
#include "../../feature_tracker/include/feature_tracker.hpp"
#include "../../landmark_server/include/landmark_server.hpp"

#include <string>

using namespace std;

int main() {
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
  cv::cvtColor( image0_origin, image0, CV_BGR2GRAY );
  cv::cvtColor( image1_origin, image1, CV_BGR2GRAY );
  cv::cvtColor( image2_origin, image2, CV_BGR2GRAY );

  FeatureTracker feature_tracker;

  bool is_projective = true;
  PoseOptimizer pose_optimizer(is_projective);

  vector<vector<cv::KeyPoint> > kp(3);
  vector<cv::Mat> desc(3);
  // matches[i] is the matches of i --> i + 1
  vector<vector<cv::DMatch> > matches(2);

  feature_tracker.DetectFeatureInFirstFrame(image0, kp[0], desc[0]);
  feature_tracker.TrackFeature(desc[0], image1, kp[1], desc[1], matches[0]);
  feature_tracker.TrackFeature(desc[1], image2, kp[2], desc[2], matches[1]);

  cv::Mat img0to1, img1to2;
  drawMatches(image0, kp[0], image1, kp[1], matches[0], img0to1,
              cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
  drawMatches(image1, kp[1], image2, kp[2], matches[1], img1to2,
              cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
  
  cv::namedWindow( "result 0 to 1", cv::WINDOW_AUTOSIZE );
  cv::imshow("result 0 to 1", img0to1); 
  cv::namedWindow( "result 1 to 2", cv::WINDOW_AUTOSIZE );
  cv::imshow("result 1 to 2", img1to2); 

  cv::waitKey(0);


  LandmarkServer landmark_server;
  landmark_server.AddFirstFrameFeature(kp[0]);
  landmark_server.AddNewFeatureAssociationToLastFrame(kp[1], matches[0]);
  landmark_server.AddNewFeatureAssociationToLastFrame(kp[2], matches[1]);
 
  landmark_server.PrintStats();

  vector<vector<cv::Vec2d> > feature_vectors(2);
  cv::Matx33d K_initial, K_final;
  vector<cv::Mat> points3d;
  vector<cv::Mat> R_ests, t_ests;

  K_initial = cv::Matx33d( 1914, 0, 640,
                           0, 1914, 360,
                           0,    0,   1);
  
  // Change data format of matched feature for initialization of 3d points.
  landmark_server.MakeFeatureVectorsForReconstruct(feature_vectors);

  pose_optimizer.initialize3DPointsFromViews(3, feature_vectors, K_initial,
                                                   points3d, R_ests, t_ests, K_final);

  return 0;
} 
