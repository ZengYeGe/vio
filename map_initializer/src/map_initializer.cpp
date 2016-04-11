#include "map_initializer.hpp"

namespace vio {

MapInitializer *MapInitializer::CreateMapInitializer(MapInitializerType type) {
  switch (type) {
    case LIVMV:
      return CreateMapInitializerLIBMV();
    case NORMALIZED8POINTFUNDAMENTAL:
      return CreateMapInitializer8Point();
    default:
      return nullptr;
  }
}

void MapInitializer::Normalize(const std::vector<cv::Vec2d> &points,
                               std::vector<cv::Vec2d> &normalized_points,
                               cv::Mat &p2norm_p) {
  // Hartley, etc, p107
  // 1. The points are translated so that their centroid is at the origin.
  // 2. The points are then scaled so that the average distance from the origin
  // is equal to sqrt(2). RMS. Root Mean Square.
  // 3. Appy on two images independently

  // Libmv uses a non-isotropic scaling. p109
  double meanX = 0.0, meanY = 0.0;
  const int num_points = points.size();

  normalized_points.resize(num_points);

  for (int i = 0; i < num_points; ++i) {
    meanX += points[i][0];
    meanY += points[i][1];
  }

  meanX = meanX / num_points;
  meanY = meanY / num_points;

  double meanDevX = 0, meanDevY = 0;

  for (int i = 0; i < num_points; ++i) {
    normalized_points[i][0] = points[i][0] - meanX;
    normalized_points[i][1] = points[i][1] - meanY;

    meanDevX += normalized_points[i][0] * normalized_points[i][0];
    meanDevY += normalized_points[i][1] * normalized_points[i][1];

    //    meanDevX += abs(normalized_points[i][0]);
    //    meanDevY += abs(normalized_points[i][1]);
  }

  meanDevX = meanDevX / num_points;
  meanDevY = meanDevY / num_points;

  //  double sX = 1.0 / meanDevX, sY = 1.0 / meanDevY;
  double sX = sqrt(2.0 / meanDevX);
  double sY = sqrt(2.0 / meanDevY);

  for (int i = 0; i < num_points; ++i) {
    normalized_points[i][0] = normalized_points[i][0] * sX;
    normalized_points[i][1] = normalized_points[i][1] * sY;
  }

  p2norm_p = cv::Mat::eye(3, 3, CV_64F);
  p2norm_p.at<double>(0, 0) = sX;
  p2norm_p.at<double>(1, 1) = sY;
  p2norm_p.at<double>(0, 2) = -meanX * sX;
  p2norm_p.at<double>(1, 2) = -meanY * sY;
}

bool MapInitializer::MakeMatrixInhomogeneous(cv::Mat &M) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      M.at<double>(i, j) = M.at<double>(i, j) / M.at<double>(2, 2);
}

}  // vio
