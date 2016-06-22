#include "vio_app.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

int TestVideo(const Options &option,
              const vio::VisualOdometryConfig &vo_config) {
  cv::VideoCapture capture(option.path);
  if (!capture.isOpened()) {
    std::cerr
        << "Failed to open the video device, video file or image sequence!\n"
        << std::endl;
    return 1;
  }

  int n = 0;
  string window_name = "video | q or esc to quit";
  std::cout << "press space to save a picture. q or esc to quit" << std::endl;
  cv::namedWindow(window_name, WINDOW_KEEPRATIO);  // resizable window;
  cv::Mat image;

  vio::VisualOdometry vo(vo_config);
  if (!vo.IsInited()) return -1;

  for (;;) {
    capture >> image;
    if (image.empty()) break;

    cv::imshow(window_name, image);
    char key = (char)waitKey(30);

    vio::FramePose current_pose;
    if (vio::ERROR == vo.TrackNewRawImage(image, current_pose)) return -1;

    switch (key) {
      case 'q':
      case 'Q':
      case 27:  // escape key
        return 0;
      default:
        break;
    }
  }

  return 0;
}
