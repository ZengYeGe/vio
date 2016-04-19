#include "vio_app.hpp"

void RunInitializer(vector<vector<cv::Vec2d> > &feature_vectors) {
  cv::Matx33d K_initial;
  vector<cv::Point3f> points3d;
  vector<bool> points3d_mask;
  vector<cv::Mat> Rs_est, ts_est;
  //  if (feature_vectors.size() == 2)
  //    K_initial = cv::Matx33d(1, 0, 0, 0, 1, 0, 0, 0, 1);
  //  else
  K_initial = cv::Matx33d(350, 0, 240, 0, 350, 360, 0, 0, 1);

  // TODO: Add option to select initializer.
  //  vio::MapInitializer *map_initializer =
  //      vio::MapInitializer::CreateMapInitializer(vio::LIVMV);
  vio::MapInitializer *map_initializer =
      vio::MapInitializer::CreateMapInitializer(
          vio::NORMALIZED8POINTFUNDAMENTAL);
  map_initializer->Initialize(feature_vectors, cv::Mat(K_initial), points3d,
                              points3d_mask, Rs_est, ts_est);

  VisualizeCamerasAndPoints(K_initial, Rs_est, ts_est, points3d);
}
