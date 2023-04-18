cmake_minimum_required(VERSION 3.1)
project(astar_cpp)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Find dependencies
FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

FIND_PACKAGE(PythonLibs 3 REQUIRED)
FIND_PACKAGE(Threads)
FIND_PACKAGE(OpenMP)

add_subdirectory(../pybind11 "${CMAKE_CURRENT_BINARY_DIR}/pybind11_build")

pybind11_add_module(astar_pybind src/astar_pybind.cpp)
set_target_properties(astar_pybind PROPERTIES PREFIX "" SUFFIX ".so")
target_include_directories(astar_pybind PRIVATE ${PYTHON_INCLUDE_DIRS}) # ${EIGEN3_INCLUDE_DIRS} 
target_link_libraries(astar_pybind PRIVATE ${PYTHON_LIBRARIES}
    Threads::Threads
    OpenMP::OpenMP_CXX)

