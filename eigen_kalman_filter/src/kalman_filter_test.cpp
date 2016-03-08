#include <iostream>
#include <Eigen/Dense>

#include "kalman_filter.hpp"

using namespace Eigen;

int main() {

  IMUKalmanFilter imu_filter;

  Matrix<double, 6, 1> imu_init_state = Matrix<double, 6, 1>::Zero();
  imu_filter.setInitialState(imu_init_state);

  Matrix<double, 6, 1> imu_measurement;
  imu_measurement << 1, 1, 1, 0.1, 0.1, 0.1;
  imu_filter.updateWithMeasurement(imu_measurement, 0.1);
  imu_filter.updateWithMeasurement(imu_measurement, 0.5);
  imu_filter.updateWithMeasurement(imu_measurement, 1.1);

  return 0;
}
