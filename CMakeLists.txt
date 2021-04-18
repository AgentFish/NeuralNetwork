#cmake --help-command-list
#cmake --help-variable-list
#cmake --help-property-list

# Initial commands
cmake_minimum_required(VERSION 3.4)
# Set project name, language & version
project(Main LANGUAGES CXX VERSION 0.1.0)
# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Generate list of source and header files
file(GLOB_RECURSE SOURCE_FILES "*.cpp")
file(GLOB_RECURSE HEADER_FILES "*.h")

# Remove files related to CMAKE build
file(GLOB_RECURSE SOURCE_CMAKE "build/*.cpp")
list(REMOVE_ITEM SOURCE_FILES ${SOURCE_CMAKE})

# Create project
add_executable(${PROJECT_NAME} ${SOURCE_FILES})