cmake_minimum_required(VERSION 3.12)
# Set CMake policies needed for pybind11
cmake_policy(SET CMP0057 NEW)  # Support IN_LIST operator
project(sqpcpu)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler optimization flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# macOS specific settings for dynamic library loading
if(APPLE)
    # Use, i.e. don't skip the full RPATH for the build tree
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    # When building, don't use the install RPATH already
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    # Add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    # Add Homebrew lib directory to RPATH
    list(APPEND CMAKE_INSTALL_RPATH "/opt/homebrew/lib")
    # Add the build directory to the RPATH
    list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR}")
    # Set the RPATH to be used when installing
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
    # Add origin to RPATH so libraries can find other libraries in the same directory
    set(CMAKE_MACOSX_RPATH ON)
endif()

# Set OsqpEigen directory explicitly
set(OsqpEigen_DIR "$ENV{HOME}/miniconda3/lib/cmake/OsqpEigen")

# Find required packages
find_package(OsqpEigen REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)

# Find pybind11 - use a more robust approach
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Define source files
set(SQPCPU_SOURCES
    src/thneed.cpp
    src/batch_thneed.cpp
)

# Create both static and shared libraries
add_library(sqpcpu_shared SHARED ${SQPCPU_SOURCES})
add_library(sqpcpu_static STATIC ${SQPCPU_SOURCES})

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

# Python bindings - use a more compatible approach
pybind11_add_module(pysqpcpu 
    src/python_bindings.cpp
)

# Link against our static library
target_link_libraries(pysqpcpu PRIVATE 
    sqpcpu_static
    ${PYTHON_LIBRARIES}
)

# Add include directories for the Python module
target_include_directories(pysqpcpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
    ${pinocchio_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${NumPy_INCLUDE_DIRS}
    "$ENV{HOME}/miniconda3/include"
)

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

# Install Python module
install(TARGETS pysqpcpu
    LIBRARY DESTINATION ${Python_SITEARCH}
)

# Optional: Build the example executables
option(BUILD_EXAMPLES "Build example executables" ON)
if(BUILD_EXAMPLES)
    # Original example
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
    
    # Batch example
    add_executable(batch_example examples/batch_example.cpp)
    target_include_directories(batch_example PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIRS}
        ${pinocchio_INCLUDE_DIRS}
        "$ENV{HOME}/miniconda3/include"
    )
    target_link_libraries(batch_example
        sqpcpu_shared  # Link against the shared library
        ${pinocchio_LIBRARIES}
        OsqpEigen::OsqpEigen
    )
    set_target_properties(batch_example PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    
    # For macOS, set the rpath for the executables
    if(APPLE)
        foreach(example_target sqpcpu_example batch_example)
            set_target_properties(${example_target} PROPERTIES
                INSTALL_RPATH "/opt/homebrew/lib;${CMAKE_INSTALL_PREFIX}/lib;${CMAKE_BINARY_DIR}"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endforeach()
    endif()
endif()