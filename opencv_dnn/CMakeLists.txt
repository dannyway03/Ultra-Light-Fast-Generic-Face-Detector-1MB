cmake_minimum_required(VERSION 3.10)

project(UltraFace)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(ultraface src/main.cpp src/cv_dnn_ultraface.cpp)

# Link libraries
target_link_libraries(ultraface ${OpenCV_LIBS})

# Set include directories for the target
target_include_directories(ultraface PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set compile options for the target
target_compile_options(ultraface PRIVATE -Wall)

# Set compile features for the target
target_compile_features(ultraface PRIVATE cxx_std_17)


# Add executable
add_executable(ultraface_speed_test src/speed_test.cpp
 src/cv_dnn_ultraface.cpp)

# Link libraries
target_link_libraries(ultraface_speed_test ${OpenCV_LIBS})

# Set include directories for the target
target_include_directories(ultraface_speed_test PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set compile options for the target
target_compile_options(ultraface_speed_test PRIVATE -Wall)

# Set compile features for the target
target_compile_features(ultraface_speed_test PRIVATE cxx_std_17)



