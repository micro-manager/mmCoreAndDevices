# CMakeLists.txt in mmCoreAndDevices

cmake_minimum_required(VERSION 3.12)
project(MicroManager)

# Optionally set the C++ standard version (change to your required version)
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add subdirectories for MMCore, MMDevice, and DeviceAdapters
add_subdirectory(MMCore)
add_subdirectory(MMDevice)
add_subdirectory(DeviceAdapters)

# Optionally add other subdirectories if needed
# add_subdirectory(OtherSubdirectory)

# Optionally, if there are any additional targets to build in this directory
# (not in the subdirectories), you can add them here

# If there are any dependencies between the targets in different subdirectories,
# you can set them up using target_link_libraries() or other CMake commands

# Optionally, you can specify the final installation location for the built targets
# using CMAKE_INSTALL_PREFIX and other related variables

# If you need additional configurations, options, or variables, you can add them here

# Optionally, you can set up the installation of the entire project using the
# install() command or other related commands, specifying where to install the
# built targets, headers, and other resources

# Optionally, you can add tests to be executed using CTest and enable_testing()

# Optionally, you can add additional build rules, custom commands, etc.

# Finally, you can set any additional properties for the entire project
# using set_project_properties() or other CMake commands

