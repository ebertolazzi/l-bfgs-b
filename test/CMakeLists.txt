cmake_minimum_required(VERSION 3.16)

project(lbfgsb_test LANGUAGES CXX)

include(FetchContent)

# Eigen 5.0.1 is provided exclusively through FetchContent.
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  # Eigen is header-only; do not configure its legacy CMake project.
  SOURCE_SUBDIR _lbfgsb_header_only
)
FetchContent_MakeAvailable(eigen)

add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE "${eigen_SOURCE_DIR}")

# Build the test suite with C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# For Windows: prevent overriding the parent project's compiler/linker settings.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(CTest)
include(GoogleTest)

add_executable(${PROJECT_NAME} src/test.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  $<$<CONFIG:Debug>:-O1;-g>
  $<$<CONFIG:MinSizeRel>:-Os>
  $<$<CONFIG:RelWithDebInfo>:-O2;-g;-march=native>
  $<$<CONFIG:Release>:-O3;-march=native>
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/../include
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  Eigen3::Eigen
  gtest_main
)

gtest_discover_tests(${PROJECT_NAME})
