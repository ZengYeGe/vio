include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(SOURCE_FILE
  src/landmark_server.cpp
)

set(TARGET "landmark_server")
add_library(${TARGET} SHARED ${SOURCE_FILE})
target_link_libraries(${TARGET}
                      ${OpenCV_LIBS})

#add_executable( landmark_server_test src/landmark_server_test.cpp )
#target_link_libraries( landmark_server_test 
#                       ${OpenCV_LIBS}
#                       landmark_server )
FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

install(FILES ${headers} 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/)
install(TARGETS ${TARGET} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )


