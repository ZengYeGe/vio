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
    std::vector<bool> &points3d_mask, std::vector<cv::Mat> &Rs,
    std::vector<cv::Mat> &ts) {
  if (feature_vectors.size() != 2) {
    std::cerr << "Error: Eight point initializer only support two views.\n";
    return false;
  }

  return InitializeTwoFrames(feature_vectors[0], feature_vectors[1], K,
                             points3d, points3d_mask, Rs, ts);
}

bool MapInitializer8Point::InitializeTwoFrames(
    const std::vector<cv::Vec2d> &kp0, const std::vector<cv::Vec2d> &kp1,
    const cv::Mat &K, std::vector<cv::Point3f> &points3d,
    std::vector<bool> &points3d_mask, std::vector<cv::Mat> &R_est,
    std::vector<cv::Mat> &t_est) {
  if (kp0.size() != kp1.size()) {
    std::cerr << "Error: keypoints number of two frames not match. Quit.\n";
    return false;
  }
  std::vector<cv::Vec2d> norm_kp0, norm_kp1;
  cv::Mat norm_T0, norm_T1;
  Normalize(kp0, norm_kp0, norm_T0);
  Normalize(kp1, norm_kp1, norm_T1);

  cv::Mat F;
  if (!ComputeFundamentalDLT(norm_kp0, norm_kp1, F)) return false;
  F = norm_T1.t() * F * norm_T0;

  // Make last element 1
  MakeMatrixInhomogeneous(F);
  if (verbose_) std::cout << "F :\n" << F << std::endl;

  /* ------------------------- Obsolete code -------------------------------
    cv::Mat F_ocv = ComputeFOpenCV(kp0, kp1);
    std::cout << "F from OpenCV: \n" << F_ocv << std::endl;

    // TODO: If K is unknown, find P
    // P1 should be [I | 0]
    cv::Mat P1, P2;
    SolveProjectionFromF(F, P1, P2);
    TriangulatePoints(kp0, kp1, P1, P2, points3d);
    std::cout << "Triangulated " << points3d.size() << " points.\n";
     ---------------------------------------------------------------------- */

  // TODO: Make tinyxml work
  // TODO: Hand pick matches for two images of calibration board?
  // TODO: Calibrate Nexus 5P

  cv::Mat E = K.t() * F * K;
  std::vector<cv::Mat> Rs(2), ts(2);
  // Recover the 4 motion hypotheses
  DecomposeE(E, Rs[0], Rs[1], ts[0]);
  ts[1] = -ts[0];

  // TODO: For now, all points are inlier.
  std::vector<bool> match_inlier_mask(kp0.size(), true);

  cv::Mat R_final, t_final;
  SelectSolutionRT(Rs, ts, K, kp0, kp1, match_inlier_mask, R_final, t_final,
                   points3d, points3d_mask);

  R_est.resize(2);
  t_est.resize(2);
  R_est[0] = cv::Mat::eye(3, 3, CV_64F);
  t_est[0] = cv::Mat::zeros(3, 1, CV_64F);
  R_final.copyTo(R_est[1]);
  t_final.copyTo(t_est[1]);

  return true;
}

template <typename Point3Type>
bool MapInitializer8Point::SelectSolutionRT(
    const std::vector<cv::Mat> &Rs, const std::vector<cv::Mat> &ts,
    const cv::Mat &K, const std::vector<cv::Vec2d> &kp0,
    const std::vector<cv::Vec2d> &kp1,
    const std::vector<bool> &match_inliers,  // not used
    cv::Mat &R_best, cv::Mat &t_best, std::vector<Point3Type> &points_3d,
    std::vector<bool> &points3d_mask) {
  std::cout << "Selecting solutions ... \n"
            << "K:\n"
            << K << std::endl;
  int max_num_point_inlier = 0;
  int best_R_id = -1, best_t_id = -1;
  for (int R_id = 0; R_id < 2; ++R_id) {
    for (int t_id = 0; t_id < 2; ++t_id) {
      std::vector<Point3Type> tmp_points3d;
      std::vector<bool> triangulated_mask;  // not used yet
      double parallax;
      // double th2 = 4.0 * 1.0;
      // double minParallax = 1.0;

      int num_point_inlier =
          EvaluateSolutionRT(Rs[R_id], ts[t_id], K, kp0, kp1, match_inliers,
                             tmp_points3d, triangulated_mask);
      std::cout << "Solution " << R_id * 2 + t_id + 1 << " has "
                << num_point_inlier << " points.\n";
      if (num_point_inlier >
          max_num_point_inlier) {  // && parallax > minParallax) {
        // Make sure there is a clear winner
        /*
                if (num_point_inlier > 0.7 * max_num_point_inlier) {
                  std::cerr < "Error: There isn't a clear winner.\n";
                  return false;
                }
        */
        max_num_point_inlier = num_point_inlier;
        best_R_id = R_id;
        best_t_id = t_id;
        points_3d = std::move(tmp_points3d);
        points3d_mask = std::move(triangulated_mask);
      }
    }
  }

  int min_triangulated = 50;
  if (max_num_point_inlier < min_triangulated) {
    std::cerr << "Not enough inlier 3D points.\n";
    return false;
  }
  Rs[best_R_id].copyTo(R_best);
  ts[best_t_id].copyTo(t_best);

  std::cout << "Best solution has " << points_3d.size() << " points.\n";

  return true;
}

bool MapInitializer8Point::ComputeFundamentalDLT(
    const std::vector<cv::Vec2d> &kp0, const std::vector<cv::Vec2d> &kp1,
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

  // Make detF = 0
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

bool MapInitializer8Point::SolveProjectionFromF(const cv::Mat &F, cv::Mat &P1,
                                                cv::Mat &P2) {
  P1 = cv::Mat::eye(3, 4, CV_64F);
  P2 = cv::Mat::zeros(3, 4, CV_64F);
  cv::Mat e2 = cv::Mat::zeros(3, 1, CV_64F);
  cv::SVD::solveZ(F.t(), e2);
  // TODO: Verify e2 is valid.
  cv::Mat P33 = P2(cv::Rect(0, 0, 3, 3));
  P33 = SkewSymmetricMatrix(e2) * F;

  e2.copyTo(P2(cv::Rect(3, 0, 1, 3)));

  if (verbose_)
    std::cout << "Compute P from F...\nP1:\n"
              << P1 << "\nP2:\n"
              << P2 << std::endl;

  return true;
}

cv::Mat MapInitializer8Point::SkewSymmetricMatrix(const cv::Mat &a) {
  cv::Mat sm(3, 3, CV_64F, cv::Scalar(0));
  sm.at<double>(0, 1) = -a.at<double>(2);
  sm.at<double>(0, 2) = a.at<double>(1);
  sm.at<double>(1, 0) = a.at<double>(2);
  sm.at<double>(1, 2) = -a.at<double>(0);
  sm.at<double>(2, 0) = -a.at<double>(1);
  sm.at<double>(2, 1) = a.at<double>(0);

  return sm;
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

template <typename Point3Type>
void MapInitializer8Point::TriangulatePoints(
    const std::vector<cv::Vec2d> &kp0, const std::vector<cv::Vec2d> &kp1,
    const cv::Mat &P1, const cv::Mat &P2, std::vector<Point3Type> &points3d) {
  points3d.resize(kp0.size());
  for (int i = 0; i < kp0.size(); ++i) {
    TriangulateDLT(kp0[i], kp1[i], P1, P2, points3d[i]);
  }
}

template <typename Point3Type>
void MapInitializer8Point::TriangulateDLT(const cv::Vec2d &kp1,
                                          const cv::Vec2d &kp2,
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

int MapInitializer8Point::EvaluateSolutionRT(
    const cv::Mat &R, const cv::Mat &t, const cv::Mat &K,
    const std::vector<cv::Vec2d> &kp0, const std::vector<cv::Vec2d> &kp1,
    const std::vector<bool> &match_inliers, std::vector<cv::Point3f> &points_3d,
    std::vector<bool> &points3d_mask) {
  if (verbose_)
    std::cout << "Evaluating solution:\n"
              << "R:\n"
              << R << "\nt:\n"
              << t << std::endl;

  // Calibration parameters
  const double fx = K.at<double>(0, 0);
  const double fy = K.at<double>(1, 1);
  const double cx = K.at<double>(0, 2);
  const double cy = K.at<double>(1, 2);

  // points_3d.resize(kp0.size());

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

  if (verbose_)
    std::cout << "P1: \n" << P1 << std::endl << "P2:\n" << P2 << std::endl;

  cv::Mat O2 = -R.t() * t;

  int nGood = 0;
  for (int i = 0; i < match_inliers.size(); ++i) {
    if (!match_inliers[i]) continue;

    cv::Mat p3dC1(3, 1, CV_64F);
    cv::Point3f point3d;
    TriangulateDLT(kp0[i], kp1[i], P1, P2, point3d);

    // TODO: Find out what's the problem
    //    Triangulate(kp0[i], kp1[i], P1, P2, p3dC1);

    // std::cout << "Point " << i << " dlt : " << point3d.x << " " << point3d.y
    // << " "
    //     << point3d.z << std::endl;
    // std::cout << "Point " << i << " lmv : " << p3dC1.at<float>(0) << " "
    //     << p3dC1.at<float>(1) << " " << p3dC1.at<float>(2) << std::endl;

    p3dC1.at<float>(0) = point3d.x;
    p3dC1.at<float>(1) = point3d.y;
    p3dC1.at<float>(2) = point3d.z;

    cv::Mat p3dC2 = R * p3dC1 + t;
    float depth1 = point3d.z;
    float depth2 = p3dC2.at<float>(2);

    if (depth1 <= 0 && depth2 <= 0) {
      points3d_mask.push_back(false);
      points_3d.push_back(cv::Point3f(0, 0, 0));
      continue;
    }

    nGood++;
    points3d_mask.push_back(true);
    points_3d.push_back(point3d);

    /*
        // TODO: Make sure isfinite is in std
        if (!std::isfinite(p3dC1.at<double>(0)) ||
            !std::isfinite(p3dC1.at<double>(1)) ||
            !std::isfinite(p3dC1.at<double>(2))) {
          vbGood[i] = false;
          continue;
        }
    */
    /*
        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        double dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        double dist2 = cv::norm(normal2);

        // std::cout << "Dist 1 " << dist1 << ", Dist 2 " << dist2 << std::endl;
        // std::cout << "3D point: " << p3dC1 << std::endl;

        double cosParallax = normal1.dot(normal2) / (dist1 * dist2);

        // std::cout << "Parallax: " << cosParallax << std::endl;
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
    */
  }
  /*
    if (nGood > 0) {
      sort(vCosParallax.begin(), vCosParallax.end());

      size_t idx = std::min(50, int(vCosParallax.size() - 1));
      parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
    } else
      parallax = 0;
  */
  return nGood;
}

}  // vio
