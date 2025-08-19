# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices (yet).

project(
    'mmdev-SequenceTester',
    'cpp',
    meson_version: '>= 1.8.3',
    default_options: [
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

mmdevice_dep = dependency('mmdevice')

boost_dep = dependency('boost')

msgpack_dep = dependency('msgpack-cxx')

sources = [
    'InterDevice.cpp',
    'LoggedSetting.cpp',
    'SequenceTester.cpp',
    'SettingLogger.cpp',
    'TextImage.cpp',
    'TriggerInput.cpp',
]

shared_module(
    'SequenceTester',
    sources,
    dependencies: [
        mmdevice_dep,
        boost_dep,
        msgpack_dep,
    ],
    gnu_symbol_visibility: 'inlineshidden',
    name_prefix: '',
    name_suffix: 'mmdev',
)
