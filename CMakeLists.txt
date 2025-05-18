cmake_minimum_required(VERSION 3.12)
project(sqpcpu)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fPIC -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC -DNDEBUG -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")


# Add the cmeel.prefix directory to CMAKE_PREFIX_PATH to find all dependencies
# list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/python3.10/dist-packages/cmeel.prefix")

# Find required packages (no need for complex path manipulations in Docker)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(hpp-fcl REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)

# Find Python and its components
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 CONFIG REQUIRED)


# Define source files
set(SQPCPU_SOURCES
src/thneed.cpp
src/batch_thneed.cpp
)

# Create shared library
add_library(sqpcpu SHARED ${SQPCPU_SOURCES})

# Include directories
target_include_directories(sqpcpu PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIRS}
${pinocchio_INCLUDE_DIRS}
${hpp-fcl_INCLUDE_DIRS}
/pinocchio/include
/usr/local/include
)
message(STATUS "Pinocchio include dirs: ${pinocchio_INCLUDE_DIRS}")

# Link libraries
target_link_libraries(sqpcpu PUBLIC
    ${pinocchio_LIBRARIES}
    OsqpEigen::OsqpEigen
    ${hpp-fcl_LIBRARIES}
)

# Set C++ standard
set_target_properties(sqpcpu PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Python bindings
pybind11_add_module(pysqpcpu src/python_bindings.cpp)
target_link_libraries(pysqpcpu PRIVATE sqpcpu)
target_include_directories(pysqpcpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Build examples
option(BUILD_EXAMPLES "Build example executables" ON)
if(BUILD_EXAMPLES)
    add_executable(sqpcpu_example examples/main.cpp)
    target_link_libraries(sqpcpu_example sqpcpu)
    
    add_executable(batch_example examples/batch_example.cpp)
    target_link_libraries(batch_example sqpcpu)
endif()