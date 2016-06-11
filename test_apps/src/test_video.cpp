#include "vio_app.hpp"

#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

int TestVideo(Options option) {

  cv::VideoCapture capture(option.path); 
  if (!capture.isOpened()) {
      std::cerr << "Failed to open the video device, video file or image sequence!\n" << std::endl;
      return 1;
  }

  int n = 0;
  string window_name = "video | q or esc to quit";
  std::cout << "press space to save a picture. q or esc to quit" << std::endl;
  cv::namedWindow(window_name, WINDOW_KEEPRATIO); //resizable window;
  cv::Mat frame;

  for (;;) {
    capture >> frame;
    if (frame.empty())
      break;

    cv::imshow(window_name, frame);
    char key = (char)waitKey(30);

    switch (key) {
      case 'q':
      case 'Q':
      case 27: //escape key
        return 0;
      default:
        break;
    }
  }

  return 0;
}
