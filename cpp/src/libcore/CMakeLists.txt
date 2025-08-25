# Rust bridge compilation
find_program(CARGO_EXECUTABLE cargo REQUIRED)
find_program(RUSTC_EXECUTABLE rustc REQUIRED)

set(RUST_BRIDGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../rust-bridge")
set(RUST_BRIDGE_TARGET_DIR "${RUST_BRIDGE_DIR}/target")

# Rust target architecture and build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_BUILD_TYPE "debug")
else()
    set(CARGO_BUILD_TYPE "release")
    set(CARGO_FLAGS "--release")
endif()

# Platform-specific lib naming
if(WIN32)
    set(POKERS_FFI_LIB "${RUST_BRIDGE_TARGET_DIR}/${CARGO_BUILD_TYPE}/pokers_ffi.lib")
    set(POKERS_FFI_LIB_NAME "pokers_ffi")
elseif(APPLE)
    set(POKERS_FFI_LIB "${RUST_BRIDGE_TARGET_DIR}/${CARGO_BUILD_TYPE}/libpokers_ffi.a")
    set(POKERS_FFI_LIB_NAME "pokers_ffi")
else()
    set(POKERS_FFI_LIB "${RUST_BRIDGE_TARGET_DIR}/${CARGO_BUILD_TYPE}/libpokers_ffi.a")
    set(POKERS_FFI_LIB_NAME "pokers_ffi")
endif()

# Build the Rust FFI library
add_custom_command(
    OUTPUT ${POKERS_FFI_LIB}
    COMMAND ${CARGO_EXECUTABLE} build ${CARGO_FLAGS}
    WORKING_DIRECTORY ${RUST_BRIDGE_DIR}
    COMMENT "Building Rust FFI library"
    VERBATIM
)

add_custom_target(pokers_ffi_build DEPENDS ${POKERS_FFI_LIB})

# Create a CMake library target for the Rust FFI library
add_library(pokers_ffi STATIC IMPORTED GLOBAL)
set_target_properties(pokers_ffi PROPERTIES IMPORTED_LOCATION ${POKERS_FFI_LIB})
add_dependencies(pokers_ffi pokers_ffi_build)

# Define the core library
add_library(deepcfr_core
    deep_cfr.cpp
    model.cpp
    pokers.cpp
)

add_dependencies(deepcfr_core pokers_ffi)

# Include directories
target_include_directories(deepcfr_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${RUST_BRIDGE_TARGET_DIR}/cxxbridge
)

# Find system libraries that Rust might need
if(UNIX AND NOT APPLE)
    find_library(PTHREAD_LIB pthread REQUIRED)
    find_library(DL_LIB dl REQUIRED)
    find_library(RT_LIB rt REQUIRED)
    set(EXTRA_LIBS ${PTHREAD_LIB} ${DL_LIB} ${RT_LIB})
else()
    set(EXTRA_LIBS "")
endif()

# Link dependencies
target_link_libraries(deepcfr_core
    PUBLIC
        ${TORCH_LIBRARIES}
        spdlog::spdlog
        Threads::Threads
        pokers_ffi
        ${EXTRA_LIBS}
)

# Set properties
set_target_properties(deepcfr_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Install rules
install(TARGETS deepcfr_core
    EXPORT deepcfr-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)