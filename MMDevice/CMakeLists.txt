# CMakeLists.txt in mmCoreAndDevices/MMDevice

# Set the source files for the MMDevice library
set(MMDEVICE_SOURCES
    Debayer.h
    DeviceBase.h
    DeviceThreads.h
    DeviceUtils.h
    FixSnprintf.h
    ImageMetadata.h
    ImgBuffer.h
    MMDevice.h
    MMDeviceConstants.h
    ModuleInterface.h
    Property.h
    Debayer.cpp
    DeviceUtils.cpp
    ImgBuffer.cpp
    MMDevice.cpp
    ModuleInterface.cpp
    Property.cpp
)

# Define the MMDevice library
add_library(MMDevice STATIC ${MMDEVICE_SOURCES})

# Include directories for libMMDevice
target_include_directories(MMDevice
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set the output directory for libMMDevice
set_target_properties(MMDevice PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Optionally install libMMDevice
if(INSTALL_MMDEVAPI)
    install(TARGETS MMDevice
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(FILES ${MMDEVICE_SOURCES}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PACKAGE}/MMDevice
    )
endif()

# Optionally add tests
add_subdirectory(unittest)
