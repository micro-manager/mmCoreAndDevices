# CMakeLists.txt in mmCoreAndDevices/MMCore

# Set the source files for the MMCore library
set(MMCORE_SOURCES
    ../MMDevice/FixSnprintf.h
    ../MMDevice/MMDevice.h
    ../MMDevice/MMDeviceConstants.h
    ../MMDevice/ModuleInterface.h
    AppleHost.h
    CircularBuffer.cpp
    CircularBuffer.h
    ConfigGroup.h
    Configuration.cpp
    Configuration.h
    CoreCallback.cpp
    CoreCallback.h
    CoreProperty.cpp
    CoreProperty.h
    CoreUtils.h
    DeviceManager.cpp
    DeviceManager.h
    Devices/AutoFocusInstance.cpp
    Devices/AutoFocusInstance.h
    Devices/CameraInstance.cpp
    Devices/CameraInstance.h
    Devices/DeviceInstance.cpp
    Devices/DeviceInstance.h
    Devices/DeviceInstances.h
    Devices/GalvoInstance.cpp
    Devices/GalvoInstance.h
    Devices/GenericInstance.h
    Devices/HubInstance.cpp
    Devices/HubInstance.h
    Devices/ImageProcessorInstance.cpp
    Devices/ImageProcessorInstance.h
    Devices/MagnifierInstance.cpp
    Devices/MagnifierInstance.h
    Devices/SLMInstance.cpp
    Devices/SLMInstance.h
    Devices/SerialInstance.cpp
    Devices/SerialInstance.h
    Devices/ShutterInstance.cpp
    Devices/ShutterInstance.h
    Devices/SignalIOInstance.cpp
    Devices/SignalIOInstance.h
    Devices/StageInstance.cpp
    Devices/StageInstance.h
    Devices/StateInstance.cpp
    Devices/StateInstance.h
    Devices/XYStageInstance.cpp
    Devices/XYStageInstance.h
    Error.cpp
    Error.h
    ErrorCodes.h
    FrameBuffer.cpp
    FrameBuffer.h
    Host.cpp
    Host.h
    LibraryInfo/LibraryPaths.h
    LibraryInfo/LibraryPathsUnix.cpp
    LoadableModules/LoadedDeviceAdapter.cpp
    LoadableModules/LoadedDeviceAdapter.h
    LoadableModules/LoadedModule.cpp
    LoadableModules/LoadedModule.h
    LoadableModules/LoadedModuleImpl.cpp
    LoadableModules/LoadedModuleImpl.h
    LoadableModules/LoadedModuleImplUnix.cpp
    LoadableModules/LoadedModuleImplUnix.h
    LogManager.cpp
    LogManager.h
    Logging/GenericStreamSink.h
    Logging/GenericEntryFilter.h
    Logging/GenericLinePacket.h
    Logging/GenericLogger.h
    Logging/GenericLoggingCore.h
    Logging/GenericMetadata.h
    Logging/GenericPacketArray.h
    Logging/GenericPacketQueue.h
    Logging/GenericSink.h
    Logging/Logger.h
    Logging/Logging.h
    Logging/Metadata.cpp
    Logging/Metadata.h
    Logging/MetadataFormatter.h
    MMCore.cpp
    MMCore.h
    PluginManager.cpp
    PluginManager.h
    Semaphore.cpp
    Semaphore.h
    Task.cpp
    Task.h
    TaskSet.cpp
    TaskSet.h
    TaskSet_CopyMemory.cpp
    TaskSet_CopyMemory.h
    ThreadPool.cpp
    ThreadPool.h
)

# Define the MMCore library
add_library(MMCore ${MMCORE_SOURCES})

# Include directories for MMCore
target_include_directories(MMCore
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../MMDevice
)

# Link MMCore with libMMDevice
target_link_libraries(MMCore PRIVATE MMDevice)

# Set the output directory for the MMCore library
set_target_properties(MMCore PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Optionally add tests
if(BUILD_CPP_TESTS)
    add_subdirectory(unittest)
endif()
