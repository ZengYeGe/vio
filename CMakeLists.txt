cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME ShuminSLAM)
project( ${PROJECT_NAME})

##############################################################################

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x") 
else() 
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lprofiler")

# TODO: Enable choose compile type.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

##############################################################################
find_package(Threads REQUIRED)
include(ExternalProject)

# Download and install GoogleTest
ExternalProject_Add(
    gtest
    URL https://googletest.googlecode.com/files/gtest-1.7.0.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    # Disable install step
    INSTALL_COMMAND ""
)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)

# Set gtest properties
ExternalProject_Get_Property(gtest source_dir binary_dir)
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
#    "INTERFACE_INCLUDE_DIRECTORIES" "${source_dir}/include"
)
# I couldn't make it work with INTERFACE_INCLUDE_DIRECTORIES
include_directories("${source_dir}/include")

##############################################################################

find_package( Ceres REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( Doxygen REQUIRED )



message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

############################################################################

add_subdirectory(util)
add_subdirectory(multiview_helper)

add_subdirectory(mapdata)
add_subdirectory(camera_model)
add_subdirectory(feature_tracker)
add_subdirectory(map_initializer)
add_subdirectory(pnp_estimator)
add_subdirectory(graph_optimizer)

add_subdirectory(visual_odometry)

add_subdirectory(kalman_filter)

add_subdirectory(test_apps)

###########################################################################

add_subdirectory(test)


###########################################################################

# Here we define the include paths that will be used by our clients.
# BUILD_INTERFACE will be used when this library is consumed in source form.
# INSTALL_INTERFACE will be used when this library is consumed in binary form.
#target_include_directories(${TARGET} PUBLIC
#	                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
#	                   $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
#	                   $<INSTALL_INTERFACE:include/${TARGET}>
#)

message(STATUS "cmake current source dir:${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "cmake current binary dir:${CMAKE_CURRENT_BINARY_DIR}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/shumin_slam.config.in
               ${CMAKE_CURRENT_BINARY_DIR}/ShuminSLAMConfig.cmake @ONLY IMMEDIATE)
export(PACKAGE ShuminSLAM)

message(STATUS "${INSTALL_CMAKE_DIR}")

