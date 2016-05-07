#include "image_frame.hpp"

#include <math.h>

namespace vio {

ImageFrame::ImageFrame(const cv::Mat &image)
      : has_grid_keypoints_(false),
        grid_width_size_(20),
        grid_height_size_(20) {
  image.copyTo(image_); 
  grid_width_index_range_ = image_.size().width / grid_width_size_;
  grid_height_index_range_ = image_.size().height / grid_height_size_;

  grid_keypoints_index_.resize(grid_width_index_range_);
  for (int i = 0; i < grid_width_index_range_; ++i)
    grid_keypoints_index_[i].resize(grid_height_index_range_); 
}
 
bool ImageFrame::CreateGridKeypointIndex() {
  if (!keypoints_.size())
    return false;

  for (int kp_id = 0; kp_id < keypoints_.size(); ++kp_id) {
    int x_grid_index = ((int) keypoints_[kp_id].pt.x) / grid_width_size_;
    int y_grid_index = ((int) keypoints_[kp_id].pt.y) / grid_height_size_;
    grid_keypoints_index_[x_grid_index][y_grid_index].push_back(kp_id);
  }
}

bool ImageFrame::GetNeighborKeypointsInRadius(cv::KeyPoint query,
                                              double dist_thresh,
                                              std::vector<cv::KeyPoint> &candidates) {
  if (!has_grid_keypoints_) return false;

  int x_grid_index = ((int) query.pt.x) / grid_width_size_;
  int y_grid_index = ((int) query.pt.y) / grid_height_size_;

  int width_search_index_range = (int) ceil(dist_thresh / grid_width_size_);
  int height_search_index_range = (int) ceil(dist_thresh / grid_height_size_);

  int min_width_index = std::max(x_grid_index - width_search_index_range, 0);
  int min_height_index = std::max(y_grid_index - height_search_index_range, 0);
  int max_width_index = std::max(x_grid_index + width_search_index_range,
                                 grid_width_index_range_);
  int max_height_index = std::max(y_grid_index + height_search_index_range,
                                  grid_height_index_range_);

  candidates.clear();
  for (int w = min_width_index; w <= max_width_index; ++w)
    for (int h = min_height_index; h <= max_height_index; ++h) {
      // TODO: Added dist threshold
      // TODO: Add unit test
      for (int i = 0; i < grid_keypoints_index_[w][h].size(); ++i)
      candidates.push_back(keypoints_[grid_keypoints_index_[w][h][i]]);
    }
  return true;
}

void ImageFrame::SetGridSize(int width, int height) {
  grid_width_size_ = width;
  grid_height_size_ = height;
}

} // vio
