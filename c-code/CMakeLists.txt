cmake_minimum_required(VERSION 2.8.11)
if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND CMAKE_POLICY)


set(PROJECT_NAME "SIFT")
project(${PROJECT_NAME})


find_package(OpenCV REQUIRED)

find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Set of helper libraries and their source codes
set(SRC_SIFT sift.cpp)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -O3")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

# Install to the source directory because I am lazy
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

if (OpenCV_FOUND)
  include_directories(
    ${PROJECT_SOURCE_DIR}
    ${OPENCV_INCLUDE_DIR}
    )

  add_library(SIFT SHARED ${SRC_SIFT})
  
  target_link_libraries(SIFT ${OpenCV_LIBRARIES})

  install(
    TARGETS SIFT
    ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}
    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}
    )

else()    
  set( OT_MISSING_DEPS "")

  if(NOT OPENCV_FOUND)
    set(OT_MISSING_DEPS "OpenCV, ${OT_MISSING_DEPS}")
  endif()

  message(STATUS "NOTICE: This project requires ${OT_MISSING_DEPS} and will not be compiled.")
endif()

