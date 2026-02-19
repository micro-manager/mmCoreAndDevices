# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices.

project(
    'MMDevice',
    'cpp',
    meson_version: '>=1.8.0',
    default_options: [
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

build_mode_args = []
if get_option('client_interface')
    build_mode_args += '-DMMDEVICE_CLIENT_BUILD'
endif

# We intentionally do NOT define NOMINMAX on Windows. MMDevice should compile
# correctly with or without Windows.h's min()/max() macros.

mmdevice_sources = files(
    'Debayer.cpp',
    'DeviceUtils.cpp',
    'ImgBuffer.cpp',
    'MMDevice.cpp',
    'ModuleInterface.cpp',
    'Property.cpp',
)

mmdevice_include_dir = include_directories('.')

mmdevice_public_headers = files(
    'Debayer.h',
    'DeviceBase.h',
    'DeviceThreads.h',
    'DeviceUtils.h',
    'ImgBuffer.h',
    'CameraImageMetadata.h',
    'MMDevice.h',
    'MMDeviceConstants.h',
    'ModuleInterface.h',
    'Property.h',
    'RegisteredDeviceCollection.h',
)
# TODO Support installing public headers

mmdevice_lib = static_library(
    'MMDevice',
    sources: mmdevice_sources,
    include_directories: mmdevice_include_dir,
    cpp_args: build_mode_args,
    # MMDevice does not depend on any external library. This is a big advantage
    # in simplifing its usage (given hundreds of device adapters depending on
    # MMDevice), so think twice before adding dependencies.
    dependencies: [
        dependency('threads'),
    ],
    gnu_symbol_visibility: 'inlineshidden',
)

# Special handling for 'tests' option. We want to:
# - Enable tests by default when built as a stand-alone project ('auto' but
#   will be enabled unless wrap fallback disabled).
# - Disable by default when built as subproject of a device adapter, because
#   we might be building many at once. But allow to enable if wanted.
# - Follow the parent project when built as subproject of mmcore ('yield: true'
#   in the option definition).
build_tests = get_option('tests').allowed()
if meson.is_subproject() and not get_option('client_interface')
    if get_option('tests').auto()
        build_tests = false
    endif
endif
if build_tests
    subdir('unittest')
endif

# Similar for 'docs' option: we don't need Doxygen docs when building a device
# adapter.
build_docs = get_option('docs').allowed()
if meson.is_subproject() and not get_option('client_interface')
    if get_option('docs').auto()
        build_tests = false
    endif
endif
if build_docs
    subdir('docs')
endif

mmdevice_dep = declare_dependency(
    compile_args: build_mode_args,
    include_directories: mmdevice_include_dir,
    link_with: mmdevice_lib,
)
meson.override_dependency('mmdevice', mmdevice_dep)

# For providing include dir to SWIG when using this project as a subproject
swig_include_dirs = [meson.current_source_dir()]
