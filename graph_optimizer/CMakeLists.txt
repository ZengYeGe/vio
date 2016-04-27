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

set(TARGET "graph_optimizer")

add_library(${TARGET} SHARED ${SOURCE_FILE})
target_link_libraries( ${TARGET} 
                       ${OpenCV_LIBS} )

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

install(FILES ${headers} 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/)
install(TARGETS ${TARGET} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )

add_executable(simple_bundle_adjuster test/simple_bundle_adjuster.cc)
target_link_libraries(simple_bundle_adjuster ceres)

find_package(Gflags)
if (GFLAGS_FOUND)
  message("-- Found Google Flags header in: ${GFLAGS_INCLUDE_DIRS}, "
          "in namespace: ${GFLAGS_NAMESPACE}")
  add_executable(libmv_bundle_adjuster
                 test/libmv_bundle_adjuster.cc)
  target_link_libraries(libmv_bundle_adjuster ceres ${GFLAGS_LIBRARIES})
else (GFLAGS_FOUND)
  message("-- Did not find Google Flags (gflags), Building without gflags "
          "- no tests or tools will be built!")
endif (GFLAGS_FOUND)
