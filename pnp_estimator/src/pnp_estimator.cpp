#include "pnp_estimator.hpp"

namespace vio {

PnPEstimator *PnPEstimator::CreatePnPEstimator(PnPMethod method) {
  switch (method) {
    // TODO: This should be OpenCV
    case ITERATIVE:
      return CreatePnPEstimatorOCV();
    default:
      return nullptr;
  }
}

}  // vio
