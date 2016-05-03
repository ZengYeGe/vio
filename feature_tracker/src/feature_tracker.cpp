#include "feature_tracker.hpp"

namespace vio {

FeatureTracker *FeatureTracker::CreateFeatureTracker(
  FeatureTrackerOptions option) {
  switch (option.method) {
    case OCV_BASIC_DETECTOR:
    case OCV_BASIC_DETECTOR_EXTRACTOR:
      return CreateFeatureTrackerOCV(option);
    case SEARCH_BY_PROJECTION:
    default:
      return nullptr;
  } 
}

} // vio
