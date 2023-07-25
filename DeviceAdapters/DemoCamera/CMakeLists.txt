# CMakeLists.txt in mmCoreAndDevices/MMDevice/DeviceAdapters/DemoCamera

# Set the target name for the device adapter library
set(TARGET_NAME mmgr_dal_DemoCamera)

# Define the sources for the device adapter library
set(SOURCES
    DemoCamera.cpp
    DemoCamera.h
    ../../MMDevice/MMDevice.h
)

link_directories(../../MMCore)
link_directories(../../lib)

# Create the device adapter library
add_library(${TARGET_NAME} MODULE ${SOURCES})

# Set the C++ standard version (change to your required version)
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_14)

# Link against external libraries (if needed)
target_link_libraries(${TARGET_NAME}
    PRIVATE
        ${MMDEVAPI_LIBADD} # Assuming MMDEVAPI_LIBADD contains external libraries
)

target_link_libraries(${TARGET_NAME} MMCore MMDevice)

# Include directories for the target
target_include_directories(${TARGET_NAME}
    PRIVATE
        ../../MMDevice/ # Add this line to include the MMDevice directory
)


