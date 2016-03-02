include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(SOURCE_FILE
  src/landmark_server.cpp
)

add_library(landmark_server ${SOURCE_FILE})

#add_executable( landmark_server_test src/landmark_server_test.cpp )
#target_link_libraries( landmark_server_test 
#                       ${OpenCV_LIBS}
#                       landmark_server )

