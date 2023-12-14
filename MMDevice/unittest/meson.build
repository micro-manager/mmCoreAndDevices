# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices.

gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_main_dep')

mmdevice_test_sources = files(
    'FloatPropertyTruncation-Tests.cpp',
    'MMTime-Tests.cpp',
)

mmdevice_test_exe = executable(
    'MMDeviceTests',
    sources: mmdevice_test_sources,
    include_directories: mmdevice_include_dir,
    link_with: mmdevice_lib,
    dependencies: gtest_dep,
)

test(
    'MMDevice unit tests',
    mmdevice_test_exe,
)