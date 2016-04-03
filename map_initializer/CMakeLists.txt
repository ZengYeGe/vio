if(Ceres_FOUND)
  add_definitions("-DCERES_FOUND=1")
  message(STATUS "Ceres is supported. Used in Landmark Initialization.")
else()
  add_definitions("-DCERES_FOUND=0")
  message(STATUS "CERES support is disabled. Ceres Solver for reconstruction API is required.")
  return()
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB SOURCE_FILE
    "include/*.hpp"
    "src/*.cpp"
)

set(TARGET "map_initializer")

add_library(${TARGET} SHARED ${SOURCE_FILE})
target_link_libraries( ${TARGET} 
                       ${OpenCV_LIBS} )

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

install(FILES ${headers} 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/)
install(TARGETS ${TARGET} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )

add_executable( map_initializer_test test/test_map_initializer.cpp)
target_link_libraries( map_initializer_test 
                       util
                       feature_tracker
                       landmark_server
                       ${TARGET}
                       ${OpenCV_LIBS} )
