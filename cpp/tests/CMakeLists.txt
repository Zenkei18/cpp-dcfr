# Set up testing
Include(FetchContent)
FetchContent_MakeAvailable(catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

# Find nlohmann/json for JSON serialization
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(json)

# Add test executable
add_executable(deepcfr_tests
    test_main.cpp
    test_model.cpp
    test_deep_cfr.cpp
)

# Link dependencies
target_link_libraries(deepcfr_tests PRIVATE
    deepcfr_core
    Catch2::Catch2WithMain
)

# Register tests with CTest
catch_discover_tests(deepcfr_tests)

# Add parity tests
add_subdirectory(parity)