#include "map_initializer.hpp"

namespace vio {

MapInitializer *MapInitializer::CreateMapInitializer(MapInitializerOptions option) {
  switch (option.method) {
    case LIVMV:
      return CreateMapInitializerLIBMV();
    case NORMALIZED8POINTFUNDAMENTAL:
      return CreateMapInitializer8Point(option);
    case ORBSLAM_F_OR_H:
      return CreateMapInitializerORBSLAM(option);
    default:
      return nullptr;
  }
}

}  // vio
