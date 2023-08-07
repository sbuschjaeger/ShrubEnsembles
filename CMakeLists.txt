cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(CShrubEnsemble LANGUAGES CXX C)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
# SET(CMAKE_CXX_EXTENSIONS OFF)
# SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)  

IF(NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release
       CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel Benchmark."
       FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -g -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native -mtune=native -fopenmp -g") 

find_package(ZLIB REQUIRED)
find_package(LZ4 REQUIRED)

add_subdirectory(lib/pybind11)

include_directories("src/se" "lib/shoco")
set_source_files_properties(lib/shoco/shoco.c PROPERTIES LANGUAGE C)

add_executable(debug lib/shoco/shoco.c tests/debug.cpp)
target_link_libraries(debug ZLIB::ZLIB LZ4::LZ4)
target_include_directories(debug PUBLIC)

# add_executable(benchmark tests/benchmark_tree.cpp)
# target_include_directories(benchmark PUBLIC)

pybind11_add_module(CShrubEnsembles src/se/PythonBindings.cpp lib/shoco/shoco.c)
