#include "pose_optimizer.hpp"

#include <iostream>

using namespace std;

PoseOptimizer::PoseOptimizer(bool is_proj)
    : is_projective_(is_proj), verbose_(true) {}

bool PoseOptimizer::initialize3DPointsFromViews(
    int num_frame, const vector<vector<cv::Vec2d>> &feature_vectors,
    const cv::Matx33d &initial_camera_matrix, vector<cv::Mat> &points3d_ests,
    vector<cv::Mat> &R_ests, vector<cv::Mat> &t_ests,
    cv::Matx33d &refined_camera_matrix) {
  if (num_frame != feature_vectors.size() || num_frame < 2) return false;

  vector<cv::Mat> all_2d_points;
  const int num_features = feature_vectors[0].size();
  for (int i = 0; i < num_frame; ++i) {
    if (num_features != feature_vectors[i].size()) return false;

    cv::Mat_<double> frame(2, num_features);
    for (int j = 0; j < num_features; ++j) {
      frame(0, j) = feature_vectors[i][j][0];
      frame(1, j) = feature_vectors[i][j][1];
    }
    all_2d_points.push_back(cv::Mat(frame));
  }

  refined_camera_matrix = cv::Mat(initial_camera_matrix).clone();
  cv::sfm::reconstruct(all_2d_points, R_ests, t_ests, refined_camera_matrix,
                       points3d_ests, is_projective_);

  if (verbose_) {
    cout << "\n----------------------------\n" << endl;
    cout << "2D feature number: " << feature_vectors[0].size() << endl;
    cout << "Initialized 3D points: " << points3d_ests.size() << endl;
    cout << "Estimated cameras: " << R_ests.size() << endl;
    cout << "Original intrinsics: " << endl
         << initial_camera_matrix << endl;
    cout << "Refined intrinsics: " << endl
         << refined_camera_matrix << endl
         << endl;
    cout << "Cameras are: " << endl;
    for (int i = 0; i < R_ests.size(); ++i) {
      cout << "R: " << endl
           << R_ests[i] << endl;
      cout << "t: " << endl
           << t_ests[i] << endl;
    }
    cout << "\n----------------------------\n" << endl;
  }

  return true;
}

bool PoseOptimizer::initialize3DPointsFromTwoViews(
    const vector<cv::KeyPoint> &kp0, const vector<cv::KeyPoint> &kp1,
    const cv::Matx33d &K, vector<cv::Point3f> &points3d_ests, cv::Mat &R,
    cv::Mat &t) {
  if (!kp0.size() || kp0.size() != kp1.size()) {
    cerr << "Error: keypoints size not match.\n";
    return false;
  }
  cv::Mat mask;

  vector<cv::Point2f> points0, points1;
  vector<cv::Point2f> normalized_p0, normalized_p1;
  cv::Mat p0_to_normalized, p1_to_normalized;

  for (int i = 0; i < kp0.size(); ++i) points0.push_back(kp0[i].pt);
  for (int i = 0; i < kp1.size(); ++i) points1.push_back(kp1[i].pt);

  normalize(points0, normalized_p0, p0_to_normalized);
  normalize(points1, normalized_p1, p1_to_normalized);

  // TODO: Write own function computing F
  cv::Mat F_norm = cv::findFundamentalMat(normalized_p0, normalized_p1,
                                          CV_FM_RANSAC, 3, 0.99, mask);
  cout << "F from normalized points: \n" << F_norm << endl;

  cv::Mat F =
      cv::findFundamentalMat(points0, points1, CV_FM_RANSAC, 3, 0.99, mask);
  cout << "F from raw        points: \n" << F << endl;

  p1_to_normalized.convertTo(p1_to_normalized, CV_64F);
  p0_to_normalized.convertTo(p0_to_normalized, CV_64F);
  cv::Mat F_recovered = p1_to_normalized.t() * F_norm * p0_to_normalized;

  // TODO: Decide which F to use.
  cout << "F from recovered  points: \n" << F_recovered << endl;

  int num_good_match = 0;
  vector<bool> good_matches(points0.size(), false);
  for (int i = 0; i < points0.size(); ++i) {
    if ((unsigned int)mask.at<uchar>(i)) {
      good_matches[i] = true;
      num_good_match++;
    }
  }

  cout << "Outlier match : " << good_matches.size() - num_good_match
       << " / " << good_matches.size() << endl;

  // ReconstructF(good_matches, F, initial_camera_matrix, R21, t21, vP3D,
  // vbTriangulated, 1.0, 50);
  vector<bool> points_3d_valid;
  return reconstructTwoViews(good_matches, kp0, kp1, F, cv::Mat(K), R,
                             t, points3d_ests, points_3d_valid, 1.0, 50);
}

// TODO: What is minParallax, minTriangulated
bool PoseOptimizer::reconstructTwoViews(
    const vector<bool> &match_inliers, const vector<cv::KeyPoint> &pts0,
    const vector<cv::KeyPoint> &pts1, const cv::Mat &F, const cv::Mat &K,
    cv::Mat &R_est, cv::Mat &t_est, vector<cv::Point3f> &points_3d,
    vector<bool> &points_3d_valid, float minParallax, int minTriangulated) {
  int num_match = 0;
  for (int i = 0; i < match_inliers.size(); ++i)
    if (match_inliers[i]) num_match++;

  cv::Mat E = K.t() * F * K;

  cv::Mat R1, R2, t;

  // Recover the 4 motion hypotheses
  decomposeE(E, R1, R2, t);

  cv::Mat t1 = t;
  cv::Mat t2 = -t;

  // Reconstruct with the 4 hyphoteses and check
  vector<cv::Point3f> points_3d1, points_3d2, points_3d3, points_3d4;
  vector<bool> points_3d_valid1, points_3d_valid2, points_3d_valid3,
      points_3d_valid4;
  float parallax1, parallax2, parallax3, parallax4;

  // TODO: make msigma2 a member
  float mSigma2 = 1.0;
  int nGood1 = CheckRT(R1, t1, pts0, pts1, match_inliers, K, points_3d1,
                       4.0 * mSigma2, points_3d_valid1, parallax1);
  int nGood2 = CheckRT(R2, t1, pts0, pts1, match_inliers, K, points_3d2,
                       4.0 * mSigma2, points_3d_valid2, parallax2);
  int nGood3 = CheckRT(R1, t2, pts0, pts1, match_inliers, K, points_3d3,
                       4.0 * mSigma2, points_3d_valid3, parallax3);
  int nGood4 = CheckRT(R2, t2, pts0, pts1, match_inliers, K, points_3d4,
                       4.0 * mSigma2, points_3d_valid4, parallax4);
  int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));

  cout << "Score of 4 solutions are: " << endl
       << "1: " << nGood1 << endl
       << "2: " << nGood2 << endl
       << "3: " << nGood3 << endl
       << "4: " << nGood4 << endl;

  R_est = cv::Mat();
  t_est = cv::Mat();

  int nMinGood = max(static_cast<int>(0.9 * num_match), minTriangulated);

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
      points_3d = points_3d1;
      points_3d_valid = points_3d_valid1;

      R1.copyTo(R_est);
      t1.copyTo(t_est);
      return true;
    }
  } else if (maxGood == nGood2) {
    if (parallax2 > minParallax) {
      points_3d = points_3d2;
      points_3d_valid = points_3d_valid2;

      R2.copyTo(R_est);
      t1.copyTo(t_est);
      return true;
    }
  } else if (maxGood == nGood3) {
    if (parallax3 > minParallax) {
      points_3d = points_3d3;
      points_3d_valid = points_3d_valid3;

      R1.copyTo(R_est);
      t2.copyTo(t_est);
      return true;
    }
  } else if (maxGood == nGood4) {
    if (parallax4 > minParallax) {
      points_3d = points_3d4;
      points_3d_valid = points_3d_valid4;

      R2.copyTo(R_est);
      t2.copyTo(t_est);
      return true;
    }
  }

  return true;
}

int PoseOptimizer::CheckRT(const cv::Mat &R, const cv::Mat &t,
                           const vector<cv::KeyPoint> &keypoints0,
                           const vector<cv::KeyPoint> &keypoints1,
                           const vector<bool> &match_inliers, const cv::Mat &K,
                           vector<cv::Point3f> &points_3d, float th2,
                           vector<bool> &vbGood, float &parallax) {
  // Calibration parameters
  const float fx = K.at<float>(0, 0);
  const float fy = K.at<float>(1, 1);
  const float cx = K.at<float>(0, 2);
  const float cy = K.at<float>(1, 2);

  vbGood = vector<bool>(keypoints0.size(), false);
  points_3d.resize(keypoints0.size());

  vector<float> vCosParallax;
  vCosParallax.reserve(keypoints0.size());

  // Camera 1 Projection Matrix K[I|0]
  cv::Mat P1(3, 4, CV_64F, cv::Scalar(0));
  K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

  cv::Mat O1 = cv::Mat::zeros(3, 1, CV_64F);

  // Camera 2 Projection Matrix K[R|t]
  cv::Mat P2(3, 4, CV_64F);
  R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
  t.copyTo(P2.rowRange(0, 3).col(3));
  P2 = K * P2;

  cout << "P1: \n" << P1 << endl << "P2:\n" << P2 << endl;

  cv::Mat O2 = -R.t() * t;

  int nGood = 0;
  for (int i = 0; i < match_inliers.size(); ++i) {
    if (!match_inliers[i]) continue;

    const cv::KeyPoint &kp1 = keypoints0[i];
    const cv::KeyPoint &kp2 = keypoints1[i];
    cv::Mat p3dC1;

    triangulate(kp1, kp2, P1, P2, p3dC1);

    if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) ||
        !isfinite(p3dC1.at<float>(2))) {
      vbGood[i] = false;
      continue;
    }

    // Check parallax
    cv::Mat normal1 = p3dC1 - O1;
    float dist1 = cv::norm(normal1);

    cv::Mat normal2 = p3dC1 - O2;
    float dist2 = cv::norm(normal2);

    float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

    // Check depth in front of first camera (only if enough parallax, as
    // "infinite" points can easily go to negative depth)
    if (p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998) continue;

    // Check depth in front of second camera (only if enough parallax, as
    // "infinite" points can easily go to negative depth)
    cv::Mat p3dC2 = R * p3dC1 + t;

    if (p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998) continue;

    // Check reprojection error in first image
    float im1x, im1y;
    float invZ1 = 1.0 / p3dC1.at<float>(2);
    im1x = fx * p3dC1.at<float>(0) * invZ1 + cx;
    im1y = fy * p3dC1.at<float>(1) * invZ1 + cy;

    float squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) +
                         (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

    if (squareError1 > th2) continue;

    // Check reprojection error in second image
    float im2x, im2y;
    float invZ2 = 1.0 / p3dC2.at<float>(2);
    im2x = fx * p3dC2.at<float>(0) * invZ2 + cx;
    im2y = fy * p3dC2.at<float>(1) * invZ2 + cy;

    float squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) +
                         (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

    if (squareError2 > th2) continue;

    vCosParallax.push_back(cosParallax);
    points_3d[i] =
        cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
    nGood++;

    if (cosParallax < 0.99998) vbGood[i] = true;
  }
  if (nGood > 0) {
    sort(vCosParallax.begin(), vCosParallax.end());

    size_t idx = min(50, int(vCosParallax.size() - 1));
    parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
  } else
    parallax = 0;
  return nGood;
}

void PoseOptimizer::triangulate(const cv::KeyPoint &kp1,
                                const cv::KeyPoint &kp2, const cv::Mat &P1,
                                const cv::Mat &P2, cv::Mat &x3D) {
  cv::Mat A(4, 4, CV_64F);

  A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
  A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
  A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
  A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

  cv::Mat u, w, vt;
  cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  x3D = vt.row(3).t();
  x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}

void PoseOptimizer::decomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2,
                               cv::Mat &t) {
  cv::Mat u, w, vt;
  cv::SVD::compute(E, w, u, vt);

  u.col(2).copyTo(t);
  t = t / cv::norm(t);

  cv::Mat W(3, 3, CV_64F, cv::Scalar(0));
  W.at<float>(0, 1) = -1;
  W.at<float>(1, 0) = 1;
  W.at<float>(2, 2) = 1;

  R1 = u * W * vt;
  if (cv::determinant(R1) < 0) R1 = -R1;

  R2 = u * W.t() * vt;
  if (cv::determinant(R2) < 0) R2 = -R2;
}

void PoseOptimizer::normalize(const vector<cv::Point2f> &points,
                              vector<cv::Point2f> &normalized_points,
                              cv::Mat &p2norm_p) {
  float meanX = 0.0, meanY = 0.0;
  const int num_points = points.size();

  normalized_points.resize(num_points);

  for (int i = 0; i < num_points; ++i) {
    meanX += points[i].x;
    meanY += points[i].y;
  }

  meanX = meanX / num_points;
  meanY = meanY / num_points;

  float meanDevX = 0, meanDevY = 0;

  for (int i = 0; i < num_points; ++i) {
    normalized_points[i].x = points[i].x - meanX;
    normalized_points[i].y = points[i].y - meanY;

    meanDevX += abs(normalized_points[i].x);
    meanDevY += abs(normalized_points[i].y);
  }

  meanDevX = meanDevX / num_points;
  meanDevY = meanDevY / num_points;

  float sX = 1.0 / meanDevX, sY = 1.0 / meanDevY;

  for (int i = 0; i < num_points; ++i) {
    normalized_points[i].x = normalized_points[i].x * sX;
    normalized_points[i].y = normalized_points[i].y * sY;
  }

  p2norm_p = cv::Mat::eye(3, 3, CV_64F);
  p2norm_p.at<float>(0, 0) = sX;
  p2norm_p.at<float>(1, 1) = sY;
  p2norm_p.at<float>(0, 2) = -meanX * sX;
  p2norm_p.at<float>(1, 2) = -meanY * sY;
}
