# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices.

gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_main_dep')

mmcore_test_sources = files(
    'APIError-Tests.cpp',
    'CoreSanity-Tests.cpp',
    'Logger-Tests.cpp',
    'LoggingSplitEntryIntoLines-Tests.cpp',
)

mmcore_test_exe = executable(
    'MMCoreTests',
    sources: mmcore_test_sources,
    include_directories: mmcore_include_dir,
    link_with: mmcore_lib,
    dependencies: [
        mmdevice_dep,
        gtest_dep,
    ],
    cpp_args: [
        '-D_CRT_SECURE_NO_WARNINGS', # TODO Eliminate the need
    ],
)

test('MMCore tests', mmcore_test_exe)