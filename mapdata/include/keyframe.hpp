#include <memory>

#include <opencv2/opencv.hpp>

#include "image_frame.hpp"

namespace vio {

class Keyframe {
 public:
  // TODO: Frame no safe
  Keyframe(std::unique_ptr<ImageFrame> frame) {
    unique_frame_id++;
    frame_id = unique_frame_id;

    image_frame_ = std::move(frame);
  }

  Keyframe() = delete;

 private:
  static int unique_frame_id;
  int frame_id;

  std::unique_ptr<ImageFrame> image_frame_;
};

int Keyframe::unique_frame_id = 0;
}  // namespace vio
