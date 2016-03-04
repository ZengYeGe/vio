include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(SOURCE_FILE
  src/feature_tracker.cpp
)

set(TARGET "feature_tracker")

add_library(${TARGET} SHARED ${SOURCE_FILE})

FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

install(FILES ${headers} 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/)
install(TARGETS ${TARGET} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )

add_executable( feature_tracker_test src/feature_tracker_test.cpp )
target_link_libraries( feature_tracker_test 
                       ${OpenCV_LIBS}
                       feature_tracker )
