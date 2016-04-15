include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package( tinyxml2 REQUIRED )

file(GLOB SOURCE_FILE
    "include/*.hpp"
    "src/*.cpp"
)

# TODO: Find other way than use as library
set(TARGET "util")

add_library(${TARGET} STATIC ${SOURCE_FILE})
