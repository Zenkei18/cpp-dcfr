# Find Python development libraries
find_package(Python3 COMPONENTS Development REQUIRED)

# Add parity test executable
add_executable(parity_test 
    parity_test_main.cpp
    state_parity_test.cpp
    agent_parity_test.cpp
)

# Link dependencies
target_link_libraries(parity_test PRIVATE
    deepcfr_core
    Catch2::Catch2WithMain
    Python3::Python
)

# Include Python directories
target_include_directories(parity_test PRIVATE
    ${Python3_INCLUDE_DIRS}
)

# Register test with CTest
add_test(NAME parity_test COMMAND parity_test)

# Copy Python test utilities
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python_utils.py 
               ${CMAKE_CURRENT_BINARY_DIR}/python_utils.py COPYONLY)
