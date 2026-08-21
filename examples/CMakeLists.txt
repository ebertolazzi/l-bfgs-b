cmake_minimum_required(VERSION 3.16)

project(lbfgsb_example LANGUAGES CXX)

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

# Build the examples with C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(${PROJECT_NAME}_constrained   src/example_constrained.cpp)
add_executable(${PROJECT_NAME}_unconstrained src/example_unconstrained.cpp)
add_executable(${PROJECT_NAME}_callback      src/example_callback.cpp)
add_executable(${PROJECT_NAME}_benchmarks    src/example_benchmarks.cpp)

foreach(_lbfgsb_target IN ITEMS
  ${PROJECT_NAME}_constrained
  ${PROJECT_NAME}_unconstrained
  ${PROJECT_NAME}_callback
  ${PROJECT_NAME}_benchmarks
)
  target_compile_options(${_lbfgsb_target} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    $<$<CONFIG:Debug>:-O1;-g>
    $<$<CONFIG:MinSizeRel>:-Os>
    $<$<CONFIG:RelWithDebInfo>:-O2;-g;-march=native>
    $<$<CONFIG:Release>:-O3;-march=native>
  )

  target_include_directories(${_lbfgsb_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/../include
    ${PROJECT_SOURCE_DIR}/include
  )

  target_link_libraries(${_lbfgsb_target} PRIVATE Eigen3::Eigen)
endforeach()
