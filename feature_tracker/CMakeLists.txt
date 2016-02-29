include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(SOURCE_FILE
  src/feature_tracker.cpp
)

add_library(feature_tracker ${SOURCE_FILE})

add_executable( feature_tracker_test src/feature_tracker_test.cpp )
target_link_libraries( feature_tracker_test 
                       ${OpenCV_LIBS}
                       feature_tracker )

