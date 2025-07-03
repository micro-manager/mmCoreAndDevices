AUTOMAKE_OPTIONS = foreign subdir-objects

AM_CPPFLAGS = -DMMDEVICE_CLIENT_BUILD

noinst_LTLIBRARIES = libMMCore.la

libMMCore_la_LIBADD = ../MMDevice/libMMDevice.la

libMMCore_la_SOURCES = \
	../MMDevice/MMDevice.h \
	../MMDevice/MMDeviceConstants.h \
	../MMDevice/ModuleInterface.h \
	CircularBuffer.cpp \
	CircularBuffer.h \
	ConfigGroup.h \
	Configuration.cpp \
	Configuration.h \
	CoreCallback.cpp \
	CoreCallback.h \
	CoreFeatures.cpp \
	CoreFeatures.h \
	CoreProperty.cpp \
	CoreProperty.h \
	CoreUtils.h \
	DeviceManager.cpp \
	DeviceManager.h \
	Devices/AutoFocusInstance.cpp \
	Devices/AutoFocusInstance.h \
	Devices/CameraInstance.cpp \
	Devices/CameraInstance.h \
	Devices/DeviceInstance.cpp \
	Devices/DeviceInstance.h \
	Devices/GenericDeviceInstance.h \
	Devices/DeviceInstances.h \
	Devices/GalvoInstance.cpp \
	Devices/GalvoInstance.h \
	Devices/GenericInstance.h \
	Devices/HubInstance.cpp \
	Devices/HubInstance.h \
	Devices/ImageProcessorInstance.cpp \
	Devices/ImageProcessorInstance.h \
	Devices/MagnifierInstance.cpp \
	Devices/MagnifierInstance.h \
	Devices/PressurePumpInstance.cpp \
	Devices/PressurePumpInstance.h \
	Devices/SLMInstance.cpp \
	Devices/SLMInstance.h \
	Devices/SerialInstance.cpp \
	Devices/SerialInstance.h \
	Devices/ShutterInstance.cpp \
	Devices/ShutterInstance.h \
	Devices/SignalIOInstance.cpp \
	Devices/SignalIOInstance.h \
	Devices/StageInstance.cpp \
	Devices/StageInstance.h \
	Devices/StateInstance.cpp \
	Devices/StateInstance.h \
	Devices/VolumetricPumpInstance.cpp \
	Devices/VolumetricPumpInstance.h \
	Devices/XYStageInstance.cpp \
	Devices/XYStageInstance.h \
	Error.cpp \
	Error.h \
	ErrorCodes.h \
	FrameBuffer.cpp \
	FrameBuffer.h \
	LibraryInfo/LibraryPaths.h \
	LibraryInfo/LibraryPathsUnix.cpp \
	LoadableModules/LoadedDeviceAdapter.cpp \
	LoadableModules/LoadedDeviceAdapter.h \
	LoadableModules/LoadedDeviceAdapterImpl.h \
	LoadableModules/LoadedDeviceAdapterImplMock.cpp \
	LoadableModules/LoadedDeviceAdapterImplMock.h \
	LoadableModules/LoadedDeviceAdapterImplRegular.cpp \
	LoadableModules/LoadedDeviceAdapterImplRegular.h \
	LoadableModules/LoadedModule.cpp \
	LoadableModules/LoadedModule.h \
	LoadableModules/LoadedModuleImpl.cpp \
	LoadableModules/LoadedModuleImpl.h \
	LoadableModules/LoadedModuleImplUnix.cpp \
	LoadableModules/LoadedModuleImplUnix.h \
	LogManager.cpp \
	LogManager.h \
	Logging/GenericStreamSink.h \
	Logging/GenericEntryFilter.h \
	Logging/GenericLinePacket.h \
	Logging/GenericLogger.h \
	Logging/GenericLoggingCore.h \
	Logging/GenericMetadata.h \
	Logging/GenericPacketArray.h \
	Logging/GenericPacketQueue.h \
	Logging/GenericSink.h \
	Logging/Logger.h \
	Logging/Logging.h \
	Logging/Metadata.cpp \
	Logging/Metadata.h \
	Logging/MetadataFormatter.h \
	MMCore.cpp \
	MMCore.h \
	MockDeviceAdapter.h \
	PluginManager.cpp \
	PluginManager.h \
	Semaphore.cpp \
	Semaphore.h \
	Task.cpp \
	Task.h \
	TaskSet.cpp \
	TaskSet.h \
	TaskSet_CopyMemory.cpp \
	TaskSet_CopyMemory.h \
	ThreadPool.cpp \
	ThreadPool.h

EXTRA_DIST = license.txt
