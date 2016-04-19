#include <memory>

#include <opencv2/opencv.hpp>

#include "image_frame.hpp"

namespace vio {

class Keyframe {
 public:
  // TODO: Frame no safe
  Keyframe(std::unique_ptr<ImageFrame> frame) {
    unique_frame_id_++;
    frame_id_ = unique_frame_id_;

    image_frame_ = std::move(frame);
  }

  Keyframe() = delete;

  int frame_id() const { return frame_id_; };

 private:
  static int unique_frame_id_;
  int frame_id_;

  std::unique_ptr<ImageFrame> image_frame_;
};

int Keyframe::unique_frame_id_ = 0;
}  // namespace vio
