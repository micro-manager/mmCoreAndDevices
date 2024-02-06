# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices.

catch2_with_main_dep = dependency(
    'catch2-with-main',
    allow_fallback: true,
    include_type: 'system',
    required: get_option('tests'),
    disabler: true,
)

mmdevice_test_sources = files(
    'DeviceUtils-Tests.cpp',
    'FloatPropertyTruncation-Tests.cpp',
    'MMTime-Tests.cpp',
)

mmdevice_test_exe = executable(
    'MMDeviceTests',
    sources: mmdevice_test_sources,
    include_directories: mmdevice_include_dir,
    link_with: mmdevice_lib,
    dependencies: catch2_with_main_dep,
)

test(
    'MMDevice unit tests',
    mmdevice_test_exe,
)
