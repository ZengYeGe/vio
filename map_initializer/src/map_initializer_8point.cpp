#include "map_initializer_8point.hpp"

#include <iostream>

namespace vio {

MapInitializer *MapInitializer::CreateMapInitializer8Point() {
  MapInitializer *initializer = new MapInitializer8Point();
  return initializer;
}

bool MapInitializer8Point::Initialize(
    const std::vector<std::vector<cv::Vec2d> > &feature_vectors,
    const cv::Mat &K, std::vector<cv::Point3f> &points3d,
    std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts) {
  if (feature_vectors.size() != 2) {
    std::cerr << "Error: Eight point initializer only support two views.\n";
    return false;
  }

  return InitializeTwoFrames(feature_vectors[0], feature_vectors[1], K,
                             points3d, Rs, ts);
}

bool MapInitializer8Point::InitializeTwoFrames(
    const std::vector<cv::Vec2d> &kp0, const std::vector<cv::Vec2d> &kp1,
    const cv::Mat &K, std::vector<cv::Point3f> &points3d,
    std::vector<cv::Mat> &Rs, std::vector<cv::Mat> &ts) {
  if (kp0.size() != kp1.size()) {
    std::cerr << "Error: keypoints number of two frames not match. Quit.\n";
    return false;
  }
  std::vector<cv::Vec2d> norm_kp0, norm_kp1;
  cv::Mat norm_T0, norm_T1;
  Normalize(kp0, norm_kp0, norm_T0);
  Normalize(kp1, norm_kp1, norm_T1);

  cv::Mat F;
  if (!ComputeF(kp0, kp1, F)) return false;

  std::cout << "F :\n" << F << std::endl;

  cv::Mat F_ocv = ComputeFOpenCV(kp0, kp1);
  std::cout << "F from OpenCV: \n" << F_ocv << std::endl;

  cv::Mat E = K.t() * F * K;
  cv::Mat R1, R2, t;
  // Recover the 4 motion hypotheses
  DecomposeE(E, R1, R2, t);

  cv::Mat t1 = t;
  cv::Mat t2 = -t;

  // Reconstruct with the 4 hyphoteses and check
  vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
  vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3,
      vbTriangulated4;
  float parallax1, parallax2, parallax3, parallax4;

  int nGood1 = EvaluateSolutionRT(R1, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers,
                       K, vP3D1, 4.0 * mSigma2, vbTriangulated1, parallax1);
  int nGood2 = EvaluateSolutionRT(R2, t1, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers,
                       K, vP3D2, 4.0 * mSigma2, vbTriangulated2, parallax2);
  int nGood3 = EvaluateSolutionRT(R1, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers,
                       K, vP3D3, 4.0 * mSigma2, vbTriangulated3, parallax3);
  int nGood4 = EvaluateSolutionRT(R2, t2, mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers,
                       K, vP3D4, 4.0 * mSigma2, vbTriangulated4, parallax4);

  int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));

  R21 = cv::Mat();
  t21 = cv::Mat();

  int nMinGood = max(static_cast<int>(0.9 * N), minTriangulated);

  int nsimilar = 0;
  if (nGood1 > 0.7 * maxGood) nsimilar++;
  if (nGood2 > 0.7 * maxGood) nsimilar++;
  if (nGood3 > 0.7 * maxGood) nsimilar++;
  if (nGood4 > 0.7 * maxGood) nsimilar++;

  // If there is not a clear winner or not enough triangulated points reject
  // initialization
  if (maxGood < nMinGood || nsimilar > 1) {
    return false;
  }

  // If best reconstruction has enough parallax initialize
  if (maxGood == nGood1) {
    if (parallax1 > minParallax) {
      vP3D = vP3D1;
      vbTriangulated = vbTriangulated1;

      R1.copyTo(R21);
      t1.copyTo(t21);
      return true;
    }
  } else if (maxGood == nGood2) {
    if (parallax2 > minParallax) {
      vP3D = vP3D2;
      vbTriangulated = vbTriangulated2;

      R2.copyTo(R21);
      t1.copyTo(t21);
      return true;
    }
  } else if (maxGood == nGood3) {
    if (parallax3 > minParallax) {
      vP3D = vP3D3;
      vbTriangulated = vbTriangulated3;

      R1.copyTo(R21);
      t2.copyTo(t21);
      return true;
    }
  } else if (maxGood == nGood4) {
    if (parallax4 > minParallax) {
      vP3D = vP3D4;
      vbTriangulated = vbTriangulated4;

      R2.copyTo(R21);
      t2.copyTo(t21);
      return true;
    }
  }

  return false;

  return true;
}

bool MapInitializer8Point::ComputeF(const std::vector<cv::Vec2d> &kp0,
                                    const std::vector<cv::Vec2d> &kp1,
                                    cv::Mat &F) {
  if (kp0.size() < 8 || kp0.size() != kp1.size()) return false;

  const int N = kp0.size();
  cv::Mat A(N, 9, CV_64F);

  for (int i = 0; i < N; ++i) {
    const double u1 = kp0[i][0];
    const double v1 = kp0[i][1];
    const double u2 = kp1[i][0];
    const double v2 = kp1[i][1];

    A.at<double>(i, 0) = u2 * u1;
    A.at<double>(i, 1) = u2 * v1;
    A.at<double>(i, 2) = u2;
    A.at<double>(i, 3) = v2 * u1;
    A.at<double>(i, 4) = v2 * v1;
    A.at<double>(i, 5) = v2;
    A.at<double>(i, 6) = u1;
    A.at<double>(i, 7) = v1;
    A.at<double>(i, 8) = 1;
  }

  cv::Mat u, w, vt;
  cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  cv::Mat Fpre = vt.row(8).reshape(0, 3);
  cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  w.at<double>(2) = 0;

  F = u * cv::Mat::diag(w) * vt;
  return true;
}

cv::Mat MapInitializer8Point::ComputeFOpenCV(
    const std::vector<cv::Vec2d> &kp0, const std::vector<cv::Vec2d> &kp1) {
  std::vector<cv::Point2f> points0(kp0.size()), points1(kp1.size());
  for (int i = 0; i < kp0.size(); ++i) {
    points0[i].x = kp0[i][0];
    points0[i].y = kp0[i][1];
    points1[i].x = kp1[i][0];
    points1[i].y = kp1[i][1];
  }
  cv::Mat mask;
  cv::Mat F = cv::findFundamentalMat(points0, points1, CV_FM_8POINT, 3, 0.99,
                                     mask);  // CV_FM_RANSAC
  return F;
}

void MapInitializer8Point::DecomposeE(const cv::Mat &E, cv::Mat &R1,
                                      cv::Mat &R2, cv::Mat &t) {
  cv::Mat u, w, vt;
  cv::SVD::compute(E, w, u, vt);

  u.col(2).copyTo(t);
  t = t / cv::norm(t);

  cv::Mat W(3, 3, CV_64F, cv::Scalar(0));
  W.at<double>(0, 1) = -1;
  W.at<double>(1, 0) = 1;
  W.at<double>(2, 2) = 1;

  R1 = u * W * vt;
  if (cv::determinant(R1) < 0) R1 = -R1;

  R2 = u * W.t() * vt;
  if (cv::determinant(R2) < 0) R2 = -R2;
}

void MapInitializer8Point::Triangulate(const cv::Vec2d &kp1,
                                       const cv::Vec2d &kp2, const cv::Mat &P1,
                                       const cv::Mat &P2, cv::Mat &x3D) {
  cv::Mat A(4, 4, CV_64F);

  A.row(0) = kp1[0] * P1.row(2) - P1.row(0);
  A.row(1) = kp1[1] * P1.row(2) - P1.row(1);
  A.row(2) = kp2[0] * P2.row(2) - P2.row(0);
  A.row(3) = kp2[1] * P2.row(2) - P2.row(1);

  cv::Mat u, w, vt;
  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}

int MapInitializer8Point::EvaluateSolutionRT(
    const cv::Mat &R, const cv::Mat &t, const std::vector<cv::Vec2d> &kp0,
    const std::vector<cv::Vec2d> &kp1, const std::vector<bool> &match_inliers,
    const cv::Mat &K, std::vector<cv::Point3f> &points_3d, double th2,
    std::vector<bool> &vbGood, double &parallax) {
  // Calibration parameters
  const double fx = K.at<double>(0, 0);
  const double fy = K.at<double>(1, 1);
  const double cx = K.at<double>(0, 2);
  const double cy = K.at<double>(1, 2);

  vbGood = std::vector<bool>(kp0.size(), false);
  points_3d.resize(kp0.size());

  std::vector<double> vCosParallax;
  vCosParallax.reserve(kp0.size());

  // Camera 1 Projection Matrix K[I|0]
  cv::Mat P1(3, 4, CV_64F, cv::Scalar(0));
  K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

  cv::Mat O1 = cv::Mat::zeros(3, 1, CV_64F);

  // Camera 2 Projection Matrix K[R|t]
  cv::Mat P2(3, 4, CV_64F);
  R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
  t.copyTo(P2.rowRange(0, 3).col(3));
  P2 = K * P2;

  cout << "P1: \n" << P1 << endl
       << "P2:\n" << P2 << endl;

  cv::Mat O2 = -R.t() * t;

  int nGood = 0;
  for (int i = 0; i < match_inliers.size(); ++i) {
    if (!match_inliers[i]) continue;

    cv::Mat p3dC1;

    Triangulate(kp0[i], kp1[i], P1, P2, p3dC1);

    // TODO: Make sure isfinite is in std
    if (!std::isfinite(p3dC1.at<double>(0)) ||
        !std::isfinite(p3dC1.at<double>(1)) ||
        !std::isfinite(p3dC1.at<double>(2))) {
      vbGood[i] = false;
      continue;
    }

    // Check parallax
    cv::Mat normal1 = p3dC1 - O1;
    double dist1 = cv::norm(normal1);

    cv::Mat normal2 = p3dC1 - O2;
    double dist2 = cv::norm(normal2);

    double cosParallax = normal1.dot(normal2) / (dist1 * dist2);

    // Check depth in front of first camera (only if enough parallax, as
    // "infinite" points can easily go to negative depth)
    if (p3dC1.at<double>(2) <= 0 && cosParallax < 0.99998) continue;

    // Check depth in front of second camera (only if enough parallax, as
    // "infinite" points can easily go to negative depth)
    cv::Mat p3dC2 = R * p3dC1 + t;

    if (p3dC2.at<double>(2) <= 0 && cosParallax < 0.99998) continue;

    // Check reprojection error in first image
    double im1x, im1y;
    double invZ1 = 1.0 / p3dC1.at<double>(2);
    im1x = fx * p3dC1.at<double>(0) * invZ1 + cx;
    im1y = fy * p3dC1.at<double>(1) * invZ1 + cy;

    double squareError1 = (im1x - kp0[i][0]) * (im1x - kp0[i][0]) +
                          (im1y - kp0[i][1]) * (im1y - kp0[i][1]);

    if (squareError1 > th2) continue;

    // Check reprojection error in second image
    double im2x, im2y;
    double invZ2 = 1.0 / p3dC2.at<double>(2);
    im2x = fx * p3dC2.at<double>(0) * invZ2 + cx;
    im2y = fy * p3dC2.at<double>(1) * invZ2 + cy;

    double squareError2 = (im2x - kp1[i][0]) * (im2x - kp1[i][0]) +
                          (im2y - kp1[i][1]) * (im2y - kp1[i][1]);

    if (squareError2 > th2) continue;

    vCosParallax.push_back(cosParallax);
    points_3d[i] = cv::Point3f(p3dC1.at<double>(0), p3dC1.at<double>(1),
                               p3dC1.at<double>(2));
    nGood++;

    if (cosParallax < 0.99998) vbGood[i] = true;
  }
  if (nGood > 0) {
    sort(vCosParallax.begin(), vCosParallax.end());

    size_t idx = std::min(50, int(vCosParallax.size() - 1));
    parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
  } else
    parallax = 0;
  return nGood;
}

}  // vio
