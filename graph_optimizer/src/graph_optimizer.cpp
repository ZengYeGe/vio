#include "graph_optimizer.hpp"

namespace vio {

GraphOptimizer *GraphOptimizer::CreateGraphOptimizer(GraphOptimizerMethod method) {
  switch (method) {
    case CERES:
      return CreateGraphOptimizerCeres();
    default:
      return nullptr;
  }
}

} // vio
