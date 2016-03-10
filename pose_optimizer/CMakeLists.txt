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

set(TARGET "pose_optimizer")
add_library(${TARGET} STATIC src/pose_optimizer.cpp)
target_link_libraries( ${TARGET} 
                       ${OpenCV_LIBS} )

FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
install(FILES ${headers} 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/)
install(TARGETS ${TARGET} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )

add_executable( pose_optimizer_test src/pose_optimizer_test.cpp )
target_link_libraries( pose_optimizer_test
                       feature_tracker
                       landmark_server
                       pose_optimizer )
