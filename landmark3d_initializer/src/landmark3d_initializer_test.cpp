#include "landmark3d_initializer.hpp"

// TODO: make the directory better
#include "../../feature_tracker/include/feature_tracker.hpp"

#include <string>

using namespace std;

int main() {
  string test_image_path =
      "/home/fan/Project/shumin_slam/landmark3d_initializer/test_data/resized_IMG_";

  string image0_name = test_image_path + "2889.jpg";
  string image1_name = test_image_path + "2890.jpg";
  string image2_name = test_image_path + "2891.jpg";

  cv::Mat image0 = cv::imread(image0_name);
  cv::Mat image1 = cv::imread(image1_name);
  cv::Mat image2 = cv::imread(image2_name);

  if (!image0.data || !image1.data || !image2.data) {
    std::cerr << "Unable to load image.\n";
    return -1;
  } 

  FeatureTracker feature_tracker;

  bool is_projective = true;
  Landmark3dInitializer landmark_initializer(is_projective);

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

  vector<vector<cv::Vec2d> > feature_vectors(2);
  cv::Matx33d K_initial, K_final;
  vector<cv::Mat> points3d;
  vector<cv::Mat> R_ests, t_ests;

  K_initial = cv::Matx33d( 1914, 0, 640,
                           0, 1914, 360,
                           0,    0,   1);
  
  // Change data format of matched feature for initialization of 3d points.

  // TODO: Make a landmark server
  // For now, only pick features that appeared in 2 frames.
  // saves each feature's id in each frame, or -1 if not seen.
  vector<vector<int> > feature_ids_in_frame; 
  for (int frame_id = 0; frame_id < 1; ++frame_id) {
    vector<int> feature_id_in_frame;
    vector<cv::Vec2d> feature_vector;
    for (int match_id = 0; match_id < matches[frame_id].size(); ++match_id) {
      int id0 = matches[frame_id][match_id].queryIdx;
      int id1 = matches[frame_id][match_id].trainIdx;
      feature_id_in_frame.push_back(id0);
      feature_id_in_frame.push_back(id1);
      
      feature_vectors[frame_id].push_back(cv::Vec2d(kp[frame_id][id0].pt.x,
                                                    kp[frame_id][id0].pt.y));
      feature_vectors[frame_id + 1].push_back(cv::Vec2d(kp[frame_id + 1][id1].pt.x,
                                                        kp[frame_id + 1][id1].pt.y));
    }
    
  }

  landmark_initializer.initialize3DPointsFromViews(2, feature_vectors, K_initial,
                                                   points3d, R_ests, t_ests, K_final);

  return 0;
} 
