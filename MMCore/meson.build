# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices.

project(
    'MMCore',
    'cpp',
    meson_version: '>=1.1.0', # May relax
    default_options: [
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

cxx = meson.get_compiler('cpp')

if cxx.get_id() in ['msvc', 'clang-cl']
    add_project_arguments('-DNOMINMAX', language: 'cpp')
endif

# MMDevice must be copied into subprojects/ for this experimental build to work
# (unless MMCore is itself being used as a subproject).
mmdevice_proj = subproject('MMDevice')
mmdevice_dep = mmdevice_proj.get_variable('mmdevice')

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
    'Devices/SerialInstance.cpp',
    'Devices/ShutterInstance.cpp',
    'Devices/SignalIOInstance.cpp',
    'Devices/SLMInstance.cpp',
    'Devices/StageInstance.cpp',
    'Devices/StateInstance.cpp',
    'Devices/XYStageInstance.cpp',
    'Error.cpp',
    'FrameBuffer.cpp',
    'LibraryInfo/LibraryPathsUnix.cpp',
    'LibraryInfo/LibraryPathsWindows.cpp',
    'LoadableModules/LoadedDeviceAdapter.cpp',
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
    'Error.h',
    'ErrorCodes.h',
    'MMCore.h',
    'MMEventCallback.h',
    # TODO MMCore.h currently includes Logging/Logger.h, which
    # should not be public. That will need to be fixed before we support
    # installing the public headers.
)

# TODO Allow MMCore to be built as a shared library, too. For that, we'd need
# to define the exported symbols on Windows (__declspec(dllexport)).
mmcore_lib = static_library(
    'MMCore',
    sources: mmcore_sources,
    include_directories: mmcore_include_dir,
    dependencies: mmdevice_dep,
    cpp_args: [
        '-D_CRT_SECURE_NO_WARNINGS', # TODO Eliminate the need
    ],
)

subdir('unittest')

mmcore = declare_dependency(
    include_directories: mmcore_include_dir,
    link_with: mmcore_lib,
    dependencies: mmdevice_dep,
)
