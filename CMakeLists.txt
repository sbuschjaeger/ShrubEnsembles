cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(CPrime LANGUAGES CXX)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
# SET(CMAKE_CXX_EXTENSIONS OFF)
# SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native -mtune=native")

add_subdirectory(lib/pybind11)

include_directories("src/prime")

add_executable(debug tests/debug.cpp)
target_include_directories(debug PUBLIC)

pybind11_add_module(CPrimeBindings src/prime/PythonBindings.cpp)
#target_include_directories(CPrimeBindings PUBLIC)
