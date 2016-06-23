file(GLOB srcs *.cpp)

add_executable(runAllTests ${srcs})
target_link_libraries(runAllTests 
                      mapdata
                      multiview_helper
                      libgtest)

add_test(name runAllTests 
         command runAllTests)
