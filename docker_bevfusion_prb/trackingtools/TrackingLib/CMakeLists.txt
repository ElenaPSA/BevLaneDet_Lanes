cmake_minimum_required(VERSION 2.6)
add_definitions(-DUSE_OPENCV)

#Configuration du projet
project(TrackingUtilsLibrary)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")


set(CMAKE_DEBUG_POSTFIX "d")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)

file(
  GLOB_RECURSE
  source_files
  src/*
  )

file(
  GLOB_RECURSE
  include_files
  include/*.hpp*
  )


include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/include")

set(TrackingUtilsLibrary_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE INTERNAL "")
add_library(TrackingUtilsLibrary ${source_files} ${include_files})

if (MSVC)
    # MSVC https://msdn.microsoft.com/en-us/library/thxezb7y.aspx
    target_compile_options(TrackingUtilsLibrary PRIVATE /W2)
else()
    # Gcc   https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
    # Clang http://clang.llvm.org/docs/UsersManual.html
    target_compile_options(TrackingUtilsLibrary PRIVATE -fPIC -Wall -Wextra -pedantic)
endif()

target_link_libraries(TrackingUtilsLibrary ${OpenCV_LIBS} )
target_include_directories(TrackingUtilsLibrary PUBLIC "./include")
install(TARGETS  TrackingUtilsLibrary DESTINATION "lib")
install(DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/include" DESTINATION ".")

