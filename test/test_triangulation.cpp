#include "gtest/gtest.h"

#include <memory>

#include "multiview.hpp"

using namespace cv;

class MultiviewTest : public ::testing::Test {
 protected:
};

TEST_F(MultiviewTest, TestTriangulation) {
  cv::FileStorage file_read;
  file_read.open("triangulation_test_data.txt", cv::FileStorage::READ);
  int kp_num = (int)file_read["NumPoints"];
  FileNode kp0node = file_read["kp0"];

  std::vector<cv::Vec2d> kp0, kp1;
  cv::Mat K, R0, t0, R1, t1;

  read(file_read["kp0"], kp0);
  read(file_read["kp1"], kp1);
  file_read["K"] >> K;
  file_read["R0"] >> R0;
  file_read["t0"] >> t0;
  file_read["R1"] >> R1;
  file_read["t1"] >> t1;

  std::vector<cv::Point3f> points3d;
  std::vector<bool> points3d_mask;

  vio::TriangulatePoints(kp0, kp1, K, R0, t0, R1, t1, points3d, points3d_mask);

  EXPECT_FALSE(true);
}
