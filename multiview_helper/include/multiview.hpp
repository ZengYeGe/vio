#include <opencv2/opencv.hpp>

namespace vio {

void RtToP(const cv::Mat &R, const cv::Mat &t, cv::Mat &P);

void TriangulatePoints(const std::vector<cv::Vec2d> &kp0,
                       const std::vector<cv::Vec2d> &kp1, const cv::Mat &K,
                       const cv::Mat &R0, const cv::Mat &t0, const cv::Mat &R1,
                       const cv::Mat &t1, std::vector<cv::Point3f> &points3d,
                       std::vector<bool> &points3d_mask);

void TriangulateDLT(const cv::Vec2d &kp1, const cv::Vec2d &kp2,
                    const cv::Mat &P1, const cv::Mat &P2, cv::Point3f &point3d);

}  // vio
