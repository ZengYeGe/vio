#include "multiview.hpp"

namespace vio {

void RtToP(const cv::Mat &R, const cv::Mat &t, cv::Mat &P) {
  P = cv::Mat(3, 4, CV_64F);
  R.copyTo(P.rowRange(0, 3).colRange(0, 3));
  t.copyTo(P.rowRange(0, 3).col(3));
}

void TriangulatePoints(const std::vector<cv::Vec2d> &kp0,
                       const std::vector<cv::Vec2d> &kp1, const cv::Mat &K,
                       const cv::Mat &R0, const cv::Mat &t0, const cv::Mat &R1,
                       const cv::Mat &t1, std::vector<cv::Point3f> &points3d,
                       std::vector<bool> &points3d_mask) {
  cv::Mat P0, P1;
  RtToP(R0, t0, P0);
  RtToP(R1, t1, P1);
  P0 = K * P0;
  P1 = K * P1;

  points3d.resize(kp0.size());
  points3d_mask.resize(kp0.size());
  for (int i = 0; i < kp0.size(); ++i) {
    TriangulateDLT(kp0[i], kp1[i], P0, P1, points3d[i]);

    cv::Mat p_global(3, 1, CV_64F);
    p_global.at<float>(0) = points3d[i].x;
    p_global.at<float>(1) = points3d[i].y;
    p_global.at<float>(2) = points3d[i].z;

    cv::Mat p3dC1 = R0 * p_global + t0;
    cv::Mat p3dC2 = R1 * p_global + t1;
    float depth1 = p3dC1.at<float>(1);
    float depth2 = p3dC2.at<float>(2);

    // TODO:
    // if (depth1 <= 0 || depth2 <= 0) {
    //  points3d_mask[i] = false;
    // } else {
    points3d_mask[i] = true;
    // }
  }
}

void TriangulateDLT(const cv::Vec2d &kp1, const cv::Vec2d &kp2,
                    const cv::Mat &P1, const cv::Mat &P2,
                    cv::Point3f &point3d) {
  cv::Mat A(4, 4, CV_64F);

  A.row(0) = kp1[0] * P1.row(2) - P1.row(0);
  A.row(1) = kp1[1] * P1.row(2) - P1.row(1);
  A.row(2) = kp2[0] * P2.row(2) - P2.row(0);
  A.row(3) = kp2[1] * P2.row(2) - P2.row(1);

  cv::Mat u, w, vt;
  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

  // It's homogeneous
  cv::Mat p3d_mat = vt.row(3).t();
  point3d.x = p3d_mat.at<double>(0) / p3d_mat.at<double>(3);
  point3d.y = p3d_mat.at<double>(1) / p3d_mat.at<double>(3);
  point3d.z = p3d_mat.at<double>(2) / p3d_mat.at<double>(3);
}

}  // vio
