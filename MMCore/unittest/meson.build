# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices.

gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_dep')

# TODO: Use a single executable for all tests -- but that requires modifying
# the sources (to remove main()), so needs to wait until we remove these tests
# from the Automake build.

mmcore_apierror_test_exe = executable(
    'MMCoreAPIErrorTest',
    sources: 'APIError-Tests.cpp',
    include_directories: mmcore_include_dir,
    link_with: mmcore_lib,
    dependencies: [
        mmdevice_dep,
        gtest_dep,
    ],
)

test('MMCore APIError test', mmcore_apierror_test_exe)

mmcore_logger_test_exe = executable(
    'MMCoreLoggerTest',
    sources: 'Logger-Tests.cpp',
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

test('MMCore Logger test', mmcore_logger_test_exe)

mmcore_loggingsplitentryintolines_test_exe = executable(
    'MMCoreLoggingSplitEntryIntoLinesTest',
    sources: 'LoggingSplitEntryIntoLines-Tests.cpp',
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

test(
    'MMCore LoggingSplitEntryIntoLines test',
    mmcore_loggingsplitentryintolines_test_exe
)
