# Find Python development libraries for parity benchmarks
find_package(Python3 COMPONENTS Development REQUIRED)

# Find nlohmann/json for JSON serialization
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(json)

# Add benchmark executable
add_executable(deepcfr_benchmarks
    bench_main.cpp
    bench_model.cpp
    bench_deep_cfr.cpp
    bench_python_parity.cpp
)

# Link dependencies
target_link_libraries(deepcfr_benchmarks PRIVATE
    deepcfr_core
    benchmark::benchmark
    Python3::Python
    nlohmann_json::nlohmann_json
)

# Include Python directories
target_include_directories(deepcfr_benchmarks PRIVATE
    ${Python3_INCLUDE_DIRS}
)

# Copy Python utility file for benchmarks
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python_utils.py 
               ${CMAKE_CURRENT_BINARY_DIR}/python_utils.py COPYONLY)

# Install benchmarks
install(TARGETS deepcfr_benchmarks
    RUNTIME DESTINATION bin/benchmarks
)