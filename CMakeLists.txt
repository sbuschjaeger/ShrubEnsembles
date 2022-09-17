cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(CShrubEnsemble LANGUAGES CXX)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
# SET(CMAKE_CXX_EXTENSIONS OFF)
# SET(CMAKE_POSITION_INDEPENDENT_CODE ON)


IF(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release
       CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel Benchmark."
       FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native -mtune=native -fopenmp -g") 

add_subdirectory(lib/pybind11)

include_directories("src/se")

# add_executable(debug tests/debug.cpp)
# target_include_directories(debug PUBLIC)

add_executable(benchmark tests/benchmark_tree.cpp)
target_include_directories(benchmark PUBLIC)

pybind11_add_module(CShrubEnsembles src/se/PythonBindings.cpp)
