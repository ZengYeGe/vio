#include <opencv2/opencv.hpp>

#include "frame.hpp"

namespace vio {

class Keyframe : public Frame {
 public:
  Keyframe(Frame &frame) : Frame(frame) {
    unique_frame_id++;
    frame_id = unique_frame_id;
  }

  Keyframe() = delete;

 private:
  static int unique_frame_id;
  int frame_id;
};

int Keyframe::unique_frame_id = 0;
}  // namespace vio
