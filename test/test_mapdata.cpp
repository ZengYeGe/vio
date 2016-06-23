#include "gtest/gtest.h"

#include <memory>

#include "image_frame.hpp"

class MapdataTest : public ::testing::Test {
 protected:
  vio::ImageFrame image_frame;
};

TEST_F(MapdataTest, TestCreateGridKeypointIndex) {
  std::vector<cv::KeyPoint> kp;

  // TODO: Add keypoints.

  image_frame.set_keypoints(kp);
}
