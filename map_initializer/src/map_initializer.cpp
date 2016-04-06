#include "map_initializer.hpp"

static MapInitializer *MapInitializer::CreateMapInitializer(MapInitializerType type) {
    switch (type) {
      case LIVMV:
        return CreateMapInitializerLIBMV();
      default:
        return nullptr;
    }
  }

void MapInitializer::Normalize(const std::vector<cv::Point2f> &points,
            std::vector<cv::Point2f> &norm_points,
            cv::Mat &p2norm_p) {
  // Hartley, etc, p107
  // 1. The points are translated so that their centroid is at the origin.
  // 2. The points are then scaled so that the average distance from the origin is equal to sqrt(2).
  // 3. Appy on two images independently

  // Libmv uses a non-isotropic scaling. p109
  float meanX = 0.0, meanY = 0.0;
  const int num_points = points.size();

  normalized_points.resize(num_points);

  for (int i = 0; i < num_points; ++i) {
    meanX += points[i].x;
    meanY += points[i].y;
  }

  meanX = meanX / num_points;
  meanY = meanY / num_points;

  float meanDevX = 0, meanDevY = 0;

  for (int i = 0; i < num_points; ++i) {
    normalized_points[i].x = points[i].x - meanX;
    normalized_points[i].y = points[i].y - meanY;

    meanDevX += abs(normalized_points[i].x);
    meanDevY += abs(normalized_points[i].y);
  }

  meanDevX = meanDevX / num_points;
  meanDevY = meanDevY / num_points;

  float sX = 1.0 / meanDevX, sY = 1.0 / meanDevY;

  for (int i = 0; i < num_points; ++i) {
    normalized_points[i].x = normalized_points[i].x * sX;
    normalized_points[i].y = normalized_points[i].y * sY;
  }

/*
  p2norm_p << sX,  0, -meanX *sX,
               0, sY, -meanY *sY,
               0,  0,          1;
*/

  p2norm_p = cv::Mat::eye(3, 3, CV_64F);
  p2norm_p.at<float>(0, 0) = sX;
  p2norm_p.at<float>(1, 1) = sY;
  p2norm_p.at<float>(0, 2) = -meanX * sX;
  p2norm_p.at<float>(1, 2) = -meanY * sY;
}
