# Find CLI11
FetchContent_GetProperties(cli11)
if(NOT cli11_POPULATED)
  FetchContent_Populate(cli11)
  add_subdirectory(${cli11_SOURCE_DIR} ${cli11_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Add training application
add_executable(train_app train.cpp)
target_link_libraries(train_app PRIVATE
    deepcfr_core
    CLI11::CLI11
)
set_target_properties(train_app PROPERTIES
    OUTPUT_NAME "deepcfr-train"
)

# Add play application
add_executable(play_app play.cpp)
target_link_libraries(play_app PRIVATE
    deepcfr_core
    CLI11::CLI11
)
set_target_properties(play_app PROPERTIES
    OUTPUT_NAME "deepcfr-play"
)

# Install applications
install(TARGETS train_app play_app
    RUNTIME DESTINATION bin
)
