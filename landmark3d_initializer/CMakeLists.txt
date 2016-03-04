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

set(TARGET "landmark3d_initializer")
add_library(${TARGET} SHARED src/landmark3d_initializer.cpp)
target_link_libraries( ${TARGET} 
                       ${OpenCV_LIBS} )

FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
install(FILES ${headers} 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/)
install(TARGETS ${TARGET} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )

add_executable( landmark3d_initializer_test src/landmark3d_initializer_test.cpp )
target_link_libraries( landmark3d_initializer_test
                       feature_tracker
                       landmark_server
                       landmark3d_initializer )
