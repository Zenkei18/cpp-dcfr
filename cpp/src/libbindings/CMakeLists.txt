pybind11_add_module(py_deepcfr MODULE bindings.cpp)

target_link_libraries(py_deepcfr PRIVATE
    deepcfr_core
    ${TORCH_LIBRARIES}
)

target_include_directories(py_deepcfr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(py_deepcfr PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "deepcfr"
)

# Install the Python module
install(TARGETS py_deepcfr
    DESTINATION ${PYTHON_SITE_PACKAGES}
)
