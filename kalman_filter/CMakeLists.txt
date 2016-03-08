cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME Kalman_filter)
project( ${PROJECT_NAME})

find_package( Eigen3 REQUIRED )

include_directories(
  ${EIGEN3_INCLUDE_DIR}
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(SOURCE_FILE
  src/kalman_filter.cpp
)

set(TARGET "kalman_filter")

add_library(${TARGET} SHARED ${SOURCE_FILE})

#FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

#install(FILES ${headers} 
#        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/)
#install(TARGETS ${TARGET} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )

add_executable( kalman_filter_test src/kalman_filter_test.cpp )
target_link_libraries( kalman_filter_test 
                       kalman_filter )
