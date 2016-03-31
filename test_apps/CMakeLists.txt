if(Ceres_FOUND)
  add_definitions("-DCERES_FOUND=1")
  message(STATUS "Ceres is supported. Used in Landmark Initialization.")
else()
  add_definitions("-DCERES_FOUND=0")
  message(STATUS "CERES support is disabled. Ceres Solver for reconstruction API is required.")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
)

add_executable( scene_reconstruction src/scene_reconstruction.cpp)
target_link_libraries( scene_reconstruction
                       ${OpenCV_LIBS} )


