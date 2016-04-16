include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB SOURCE_FILE
    "include/*.hpp"
    "src/*.cpp"
)

#message(STATUS "${SOURCE_FILE}")

#set (SOURCE_FILE
#    src/feature_matcher_ocv.cpp
#    src/feature_tracker_ocv.cpp
#}

set(TARGET "feature_tracker")

add_library(${TARGET} SHARED ${SOURCE_FILE})
target_link_libraries( ${TARGET} 
                       ${OpenCV_LIBS} )

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

install(FILES ${headers} 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/)
install(TARGETS ${TARGET} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )

add_executable( feature_tracker_test test/test_feature_tracker.cpp)
target_link_libraries( feature_tracker_test 
                       util
                       mapdata
                       ${TARGET}
                       ${OpenCV_LIBS} )
