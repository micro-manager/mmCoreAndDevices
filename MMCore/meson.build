# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices.

project(
    'MMCore',
    'cpp',
    meson_version: '>=1.8.0',
    default_options: [
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

cxx = meson.get_compiler('cpp')

if cxx.get_id() in ['msvc', 'clang-cl']
    add_project_arguments('-DNOMINMAX', language: 'cpp')
endif

# mmdevice is provided via a .wrap, but it should be manually copied into
# subprojects in order to control the version that gets fetched.
mmdevice_proj = subproject(
    'mmdevice',
    default_options: {
        'client_interface': true,  # Build for use by mmcore
    },
)
mmdevice_dep = mmdevice_proj.get_variable('mmdevice_dep')

mmcore_sources = files(
    'CircularBuffer.cpp',
    'Configuration.cpp',
    'CoreCallback.cpp',
    'CoreFeatures.cpp',
    'CoreProperty.cpp',
    'DeviceManager.cpp',
    'Devices/AutoFocusInstance.cpp',
    'Devices/CameraInstance.cpp',
    'Devices/DeviceInstance.cpp',
    'Devices/GalvoInstance.cpp',
    'Devices/HubInstance.cpp',
    'Devices/ImageProcessorInstance.cpp',
    'Devices/MagnifierInstance.cpp',
    'Devices/PressurePumpInstance.cpp',
    'Devices/SerialInstance.cpp',
    'Devices/ShutterInstance.cpp',
    'Devices/SignalIOInstance.cpp',
    'Devices/SLMInstance.cpp',
    'Devices/StageInstance.cpp',
    'Devices/StateInstance.cpp',
    'Devices/VolumetricPumpInstance.cpp',
    'Devices/XYStageInstance.cpp',
    'Error.cpp',
    'FrameBuffer.cpp',
    'LibraryInfo/LibraryPathsUnix.cpp',
    'LibraryInfo/LibraryPathsWindows.cpp',
    'LoadableModules/LoadedDeviceAdapter.cpp',
    'LoadableModules/LoadedDeviceAdapterImplMock.cpp',
    'LoadableModules/LoadedDeviceAdapterImplRegular.cpp',
    'LoadableModules/LoadedModule.cpp',
    'LoadableModules/LoadedModuleImpl.cpp',
    'LoadableModules/LoadedModuleImplUnix.cpp',
    'LoadableModules/LoadedModuleImplWindows.cpp',
    'Logging/Metadata.cpp',
    'LogManager.cpp',
    'MMCore.cpp',
    'PluginManager.cpp',
    'Semaphore.cpp',
    'Task.cpp',
    'TaskSet.cpp',
    'TaskSet_CopyMemory.cpp',
    'ThreadPool.cpp',
)

mmcore_include_dir = include_directories('.')

mmcore_public_headers = files(
    'Configuration.h',
    'CoreDeclHelpers.h',
    'Error.h',
    'ErrorCodes.h',
    'ImageMetadata.h',
    'Logging/GenericLogger.h',
    'Logging/Logger.h',
    'Logging/Metadata.h',
    'Logging/GenericMetadata.h',
    'MMCore.h',
    'MMEventCallback.h',
    'MockDeviceAdapter.h',
)
# Note that the MMDevice headers are also needed; which of those are part of
# MMCore's public interface is poorly defined at the moment.

# TODO Allow MMCore to be built as a shared library, too. For that, we'd need
# to define the exported symbols on Windows (__declspec(dllexport)).
mmcore_lib = static_library(
    'MMCore',
    sources: mmcore_sources,
    include_directories: mmcore_include_dir,
    dependencies: [
        mmdevice_dep,
        dependency('threads'),
        dependency('dl', required: false),
    ],
    gnu_symbol_visibility: 'inlineshidden',
    cpp_args: [
        '-D_CRT_SECURE_NO_WARNINGS', # TODO Eliminate the need
    ],
)

subdir('unittest')

subdir('docs')

mmcore_dep = declare_dependency(
    include_directories: mmcore_include_dir,
    link_with: mmcore_lib,
    dependencies: mmdevice_dep,
)
meson.override_dependency('mmcore', mmcore_dep)

# For providing include dir to SWIG when using this project as a subproject
swig_include_dirs = mmdevice_proj.get_variable('swig_include_dirs') + [
    meson.current_source_dir(),
]
