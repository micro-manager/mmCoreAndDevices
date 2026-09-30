# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices (yet).

project(
    'mmdev-Utilities',
    'cpp',
    meson_version: '>= 1.8.3',
    default_options: [
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

mmdevice_dep = dependency('mmdevice')

sources = [
    'AutoFocusStage.cpp',
    'ComboXYStage.cpp',
    'DAGalvo.cpp',
    'DAMonochromator.cpp',
    'DAShutter.cpp',
    'DATTLStateDevice.cpp',
    'DAXYStage.cpp',
    'DAZStage.cpp',
    'MultiCamera.cpp',
    'MultiDAStateDevice.cpp',
    'MultiShutter.cpp',
    'MultiStage.cpp',
    'PropertyShutter.cpp',
    'SerialDTRShutter.cpp',
    'SingleAxisStage.cpp',
    'StateDeviceShutter.cpp',
    'Utilities.cpp',
]

shared_module(
    'Utilities',
    sources,
    dependencies: [
        mmdevice_dep,
    ],
    gnu_symbol_visibility: 'inlineshidden',
    name_prefix: '',
    name_suffix: 'mmdev',
)
