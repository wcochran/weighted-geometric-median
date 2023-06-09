cmake_minimum_required(VERSION 3.19)
project( geometric-median )
set( CMAKE_CXX_STANDARD 17 )

#find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Eigen3 REQUIRED NO_MODULE)

add_compile_options(-Wno-unknown-warning-option)

set (CMAKE_CXX_FLAGS "-DTEST_WEIGHTED_GEOMETRIC_MEAN")

add_executable( geometric-median geometric-median.cpp geometric-median.h )
target_link_libraries( geometric-median Eigen3::Eigen )


