set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lprofiler")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB SOURCE_FILE
    "include/*.hpp"
    "src/*.cpp"
)

message(STATUS "${SOURCE_FILE}")

set(TARGET "visual_odometry")

add_library(${TARGET} SHARED ${SOURCE_FILE})
target_link_libraries( ${TARGET} 
                       ${OpenCV_LIBS}
                       camera_model
                       util
                       multiview_helper
                       mapdata
                       feature_tracker
                       graph_optimizer
                       pnp_estimator
                       map_initializer)

target_include_directories(${TARGET} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
