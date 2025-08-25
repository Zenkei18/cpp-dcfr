cmake_minimum_required(VERSION 3.16)
project(deepcfr-cpp VERSION 0.1.0 LANGUAGES CXX)

# Set C++20 as the default standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(DEEPCFR_BUILD_TESTS "Build test suite" ON)
option(DEEPCFR_BUILD_BENCHMARKS "Build benchmarks" ON)
option(DEEPCFR_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# Find dependencies
find_package(Threads REQUIRED)
find_package(spdlog REQUIRED)

# Include FetchContent for downloading missing dependencies
include(FetchContent)

# Add Eigen for matrix operations
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)

# Add pybind11 for Python bindings
if(DEEPCFR_BUILD_PYTHON_BINDINGS)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Add Catch2 for testing
if(DEEPCFR_BUILD_TESTS)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(catch2)
endif()

# Add Google Benchmark
if(DEEPCFR_BUILD_BENCHMARKS)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.2
        GIT_SHALLOW TRUE
        CMAKE_ARGS "-DBENCHMARK_ENABLE_TESTING=OFF"
    )
    FetchContent_MakeAvailable(googlebenchmark)
endif()

# Add CLI11 for command line parsing
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.3.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cli11)

# Set up include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
)

# Find LibTorch (PyTorch C++ API)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/libtorch)
find_package(Torch REQUIRED)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add library targets
add_subdirectory(src/libcore)

# Add Python bindings if enabled
if(DEEPCFR_BUILD_PYTHON_BINDINGS)
    add_subdirectory(src/libbindings)
endif()

# Add applications
add_subdirectory(src/apps)

# Add tests if enabled
if(DEEPCFR_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Add benchmarks if enabled
if(DEEPCFR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation rules
install(DIRECTORY include/deepcfr DESTINATION include)

# Generate package config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/deepcfr-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/deepcfr-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/deepcfr-config-version.cmake"
    DESTINATION lib/cmake/deepcfr
)

# Create a simple config file
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/deepcfr-config.cmake"
    "include(\${CMAKE_CURRENT_LIST_DIR}/deepcfr-targets.cmake)\n"
)

# Export targets
install(
    EXPORT deepcfr-targets
    NAMESPACE deepcfr::
    DESTINATION lib/cmake/deepcfr
)
