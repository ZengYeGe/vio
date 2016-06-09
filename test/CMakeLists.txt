file(GLOB srcs *.cpp)

add_executable(runAllTests ${srcs})
target_link_libraries(runAllTests 
                      mapdata
                      libgtest)

add_test(name runAllTests 
         command runAllTests)
