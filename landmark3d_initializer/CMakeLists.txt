if(Ceres_FOUND)
  add_definitions("-DCERES_FOUND=1")
  message(STATUS "Ceres is supported. Used in Landmark Initialization.")
else()
  add_definitions("-DCERES_FOUND=0")
  message(STATUS "CERES support is disabled. Ceres Solver for reconstruction API is required.")
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(landmark3d_initializer src/landmark3d_initializer.cpp)

add_executable( landmark3d_initializer_test src/landmark3d_initializer_test.cpp )

target_link_libraries( landmark3d_initializer
                       ${OpenCV_LIBS} )

target_link_libraries( landmark3d_initializer_test
                       feature_tracker
                       landmark3d_initializer )
