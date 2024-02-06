# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices.

project(
    'MMCore',
    'cpp',
    meson_version: '>=1.2.0',
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
if get_option('tests').enabled()
    tests_option = 'enabled'
elif get_option('tests').disabled()
    tests_option = 'disabled'
else
    tests_option = 'auto'
endif
mmdevice_proj = subproject(
    'MMDevice',
    default_options: {
        'client_interface': true,
        # Propagate value of 'tests' option ('yield: true' in MMDeivce's
        # 'tests' option did not seem to work; Meson 1.3.1).
        'tests': tests_option,
    },
)
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
    'Logging/GenericLogger.h',
    'Logging/Logger.h',
    'Logging/Metadata.h',
    'Logging/GenericMetadata.h',
    'MMCore.h',
    'MMEventCallback.h',
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
    ],
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

# For providing include dir to SWIG when using this project as a subproject
swig_include_dirs = mmdevice_proj.get_variable('swig_include_dirs') + [
    meson.current_source_dir(),
]
