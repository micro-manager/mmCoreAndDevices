# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices.

catch2_with_main_dep = dependency(
    'catch2-with-main',
    allow_fallback: true,
    include_type: 'system',
    required: get_option('tests'),
    disabler: true,
)

mmcore_test_sources = files(
    'APIError-Tests.cpp',
    'CoreCreateDestroy-Tests.cpp',
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
        catch2_with_main_dep,
    ],
    cpp_args: [
        '-D_CRT_SECURE_NO_WARNINGS', # TODO Eliminate the need
    ],
)

test('MMCore tests', mmcore_test_exe)
