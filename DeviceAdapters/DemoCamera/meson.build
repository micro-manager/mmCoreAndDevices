# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices (yet).

project(
    'mmdev-DemoCamera',
    'cpp',
    meson_version: '>= 1.8.3',
    default_options: [
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

cxx = meson.get_compiler('cpp')

if cxx.get_id() in ['msvc', 'clang-cl']
    add_project_arguments('-DNOMINMAX', language: 'cpp')
endif

mmdevice_dep = dependency('mmdevice')

winmm_lib = cxx.find_library('Winmm', required: false)

sources = [
    'DemoCamera.cpp',
]

shared_module(
    'DemoCamera',
    sources,
    dependencies: [
        mmdevice_dep,
        winmm_lib,
    ],
    gnu_symbol_visibility: 'inlineshidden',
    name_prefix: '',
    name_suffix: 'mmdev',
)
