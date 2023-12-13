# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices.

gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_dep')

# TODO: Use a single executable for all tests -- but that requires modifying
# the sources (to remove main()), so needs to wait until we remove these tests
# from the Automake build.

mmdevice_floatpropertytruncation_test_exe = executable(
    'MMDeviceFloatPropertyTruncationTest',
    sources: 'FloatPropertyTruncation-Tests.cpp',
    include_directories: mmdevice_include_dir,
    link_with: mmdevice_lib,
    dependencies: gtest_dep,
)

test(
    'MMDevice FloatPropertyTruncation test',
    mmdevice_floatpropertytruncation_test_exe,
)

mmdevice_mmtime_test_exe = executable(
    'MMDeviceMMTimeTest',
    sources: 'MMTime-Tests.cpp',
    include_directories: mmdevice_include_dir,
    link_with: mmdevice_lib,
    dependencies: gtest_dep,
)

test('MMDevice MMTime test', mmdevice_mmtime_test_exe)
