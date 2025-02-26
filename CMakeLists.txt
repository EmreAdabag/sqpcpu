cmake_minimum_required(VERSION 3.0)
project(sqpcpu)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler optimization flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# Set OsqpEigen directory explicitly
set(OsqpEigen_DIR "$ENV{HOME}/miniconda3/lib/cmake/OsqpEigen")

# Find required packages
find_package(OsqpEigen REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)

# Create both static and shared libraries
add_library(sqpcpu_shared SHARED src/thneed.cpp)
add_library(sqpcpu_static STATIC src/thneed.cpp)

# Set the output names to avoid conflicts
set_target_properties(sqpcpu_shared PROPERTIES 
    OUTPUT_NAME sqpcpu
    VERSION 1.0.0
    SOVERSION 1)
set_target_properties(sqpcpu_static PROPERTIES 
    OUTPUT_NAME sqpcpu)

# Include directories for both libraries
foreach(target sqpcpu_shared sqpcpu_static)
    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIRS}
        ${pinocchio_INCLUDE_DIRS}
        "$ENV{HOME}/miniconda3/include"
    )

    # Link libraries
    target_link_libraries(${target}
        ${pinocchio_LIBRARIES}
        OsqpEigen::OsqpEigen
    )

    # Set C++ standard
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endforeach()

# Install rules
include(GNUInstallDirs)
install(TARGETS sqpcpu_shared sqpcpu_static
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install header files
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Optional: Build the example executable
option(BUILD_EXAMPLES "Build example executables" ON)
if(BUILD_EXAMPLES)
    add_executable(sqpcpu_example examples/main.cpp)
    target_include_directories(sqpcpu_example PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIRS}
        ${pinocchio_INCLUDE_DIRS}
        "$ENV{HOME}/miniconda3/include"
    )
    target_link_libraries(sqpcpu_example
        sqpcpu_shared  # Link against the shared library
        ${pinocchio_LIBRARIES}
        OsqpEigen::OsqpEigen
    )
    set_target_properties(sqpcpu_example PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endif()