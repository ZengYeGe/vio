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
    float depth1 = p3dC1.at<float>(2);
    float depth2 = p3dC2.at<float>(2);

    // TODO:
    if (depth1 <= 0 || depth2 <= 0) {
      points3d_mask[i] = false;
    } else {
      points3d_mask[i] = true;
    }
  }
}

template <typename Point3Type>
void TriangulateDLT(const cv::Vec2d &kp1, const cv::Vec2d &kp2,
                    const cv::Mat &P1, const cv::Mat &P2,
                    Point3Type &point3d) {
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

void Normalize(const std::vector<cv::Vec2d> &points,
                               std::vector<cv::Vec2d> &normalized_points,
                               cv::Mat &p2norm_p) {
  // Hartley, etc, p107
  // 1. The points are translated so that their centroid is at the origin.
  // 2. The points are then scaled so that the average distance from the origin
  // is equal to sqrt(2). RMS. Root Mean Square.
  // 3. Appy on two images independently

  // Libmv uses a non-isotropic scaling. p109
  double meanX = 0.0, meanY = 0.0;
  const int num_points = points.size();

  normalized_points.resize(num_points);

  for (int i = 0; i < num_points; ++i) {
    meanX += points[i][0];
    meanY += points[i][1];
  }

  meanX = meanX / num_points;
  meanY = meanY / num_points;

  double meanDevX = 0, meanDevY = 0;

  for (int i = 0; i < num_points; ++i) {
    normalized_points[i][0] = points[i][0] - meanX;
    normalized_points[i][1] = points[i][1] - meanY;

    meanDevX += normalized_points[i][0] * normalized_points[i][0];
    meanDevY += normalized_points[i][1] * normalized_points[i][1];

    //    meanDevX += abs(normalized_points[i][0]);
    //    meanDevY += abs(normalized_points[i][1]);
  }

  meanDevX = meanDevX / num_points;
  meanDevY = meanDevY / num_points;

  //  double sX = 1.0 / meanDevX, sY = 1.0 / meanDevY;
  double sX = sqrt(2.0 / meanDevX);
  double sY = sqrt(2.0 / meanDevY);

  for (int i = 0; i < num_points; ++i) {
    normalized_points[i][0] = normalized_points[i][0] * sX;
    normalized_points[i][1] = normalized_points[i][1] * sY;
  }

  p2norm_p = cv::Mat::eye(3, 3, CV_64F);
  p2norm_p.at<double>(0, 0) = sX;
  p2norm_p.at<double>(1, 1) = sY;
  p2norm_p.at<double>(0, 2) = -meanX * sX;
  p2norm_p.at<double>(1, 2) = -meanY * sY;
}

bool MakeMatrixInhomogeneous(cv::Mat &M) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      M.at<double>(i, j) = M.at<double>(i, j) / M.at<double>(2, 2);
}


bool SolveProjectionFromF(const cv::Mat &F, cv::Mat &P1,
                                                cv::Mat &P2) {
  P1 = cv::Mat::eye(3, 4, CV_64F);
  P2 = cv::Mat::zeros(3, 4, CV_64F);
  cv::Mat e2 = cv::Mat::zeros(3, 1, CV_64F);
  cv::SVD::solveZ(F.t(), e2);
  // TODO: Verify e2 is valid.
  cv::Mat P33 = P2(cv::Rect(0, 0, 3, 3));
  P33 = SkewSymmetricMatrix(e2) * F;

  e2.copyTo(P2(cv::Rect(3, 0, 1, 3)));

  std::cout << "Compute P from F...\nP1:\n"
            << P1 << "\nP2:\n"
            << P2 << std::endl;

  return true;
}

cv::Mat SkewSymmetricMatrix(const cv::Mat &a) {
  cv::Mat sm(3, 3, CV_64F, cv::Scalar(0));
  sm.at<double>(0, 1) = -a.at<double>(2);
  sm.at<double>(0, 2) = a.at<double>(1);
  sm.at<double>(1, 0) = a.at<double>(2);
  sm.at<double>(1, 2) = -a.at<double>(0);
  sm.at<double>(2, 0) = -a.at<double>(1);
  sm.at<double>(2, 1) = a.at<double>(0);

  return sm;
}

}  // vio
