#include "vio_app.hpp"

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

  std::unique_ptr<vio::ImageFrame> last_frame(new vio::ImageFrame(image0));
  std::unique_ptr<vio::ImageFrame> last_frame_key(new vio::ImageFrame(image0));

  feature_tracker->TrackFirstFrame(*last_frame);
  feature_tracker->TrackFirstFrame(*last_frame_key);
  std::cout << "Found " << last_frame->keypoints().size() << " features.\n";

  KeyframeSelector keyframe_selector;

  vio::Map vio_map;
  // TODO: Doesn't make sense to do move a lot
  std::unique_ptr<vio::Keyframe> first_keyframe(
      new vio::Keyframe(std::move(last_frame_key)));
  vio_map.AddFirstKeyframe(std::move(first_keyframe));

  cv::namedWindow("tracking_result", cv::WINDOW_AUTOSIZE);
  int num_frames = 0;
  for (int i = 1; i < images.size(); ++i) {
    cv::Mat image1 = cv::imread(images[i]);
    if (!image1.data) {
      cerr << "Error: Unable to load image " << images[i] << endl;
      return -1;
    }
    std::unique_ptr<vio::ImageFrame> new_frame(new vio::ImageFrame(image1));

    std::vector<cv::DMatch> matches;
    feature_tracker->TrackFrame(vio_map.GetLastKeyframe().image_frame(),
                                *new_frame, matches);

    std::cout << "Found " << matches.size() << " matches.\n";

    cv::Mat output_img = new_frame->GetImage().clone();

    int thickness = 2;
    for (int i = 0; i < matches.size(); ++i) {
      line(output_img, new_frame->keypoints()[matches[i].trainIdx].pt,
           last_frame->keypoints()[matches[i].queryIdx].pt,
           cv::Scalar(255, 0, 0), thickness);
    }

    cv::imshow("tracking_result", output_img);
    cv::waitKey(0);

    if (option.use_keyframe) {
      if (!keyframe_selector.isKeyframe(matches)) continue;
    }

    std::unique_ptr<vio::Keyframe> new_keyframe(
        new vio::Keyframe(std::move(new_frame)));
    vio_map.AddNewKeyframeMatchToLastKeyframe(std::move(new_keyframe), matches);

    last_frame = std::move(new_frame);
    num_frames++;
  }

  vector<vector<cv::Vec2d> > feature_vectors(images.size());
  vio_map.PrepareInitializationData(feature_vectors);

  RunInitializer(feature_vectors);

  return 0;
}
