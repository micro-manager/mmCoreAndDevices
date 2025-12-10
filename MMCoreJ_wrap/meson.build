# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices.

# IMPORTANT: This build script uses mmcore and mmdevice from the mirror
# repositories, not the MMCore/ and MMDevice/ in mmCoreAndDevices. This is
# because this build script is expected to be put in use when MMCoreJ is moved
# out of mmCoreAndDevices.
# The versions used are defined in mmdevice.wrap and mmcore.wrap.

project(
    'mmcorej',
    'cpp', 'java',
    # TODO version (set here and generate pom.xml)
    meson_version: '>=1.3.0',
    default_options: [
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

cxx = meson.get_compiler('cpp')

if cxx.get_id() in ['gcc', 'clang']
    # SWIG Java typemaps call for -fno-strict-aliasing when compiling the
    # SWIG-generated code (see SWIG docs).
    add_project_arguments('-fno-strict-aliasing', language: 'cpp')
endif

if cxx.get_id() in ['msvc', 'clang-cl']
    add_project_arguments('-DNOMINMAX', language: 'cpp')
endif

fs = import('fs')

swig = find_program('swig', native: true)

mmcore_proj = subproject(
    'mmcore',
    default_options: {
        'default_library': 'static',
        'tests': 'disabled',
    },
)
mmcore_dep = mmcore_proj.get_variable('mmcore')

jni_dep = dependency('jni', version: '>= 1.8.0', modules: ['jvm'])

threads_dep = dependency('threads')

swig_include_dirs = mmcore_proj.get_variable('swig_include_dirs')
swig_incdir_args = []
foreach abspath : swig_include_dirs
    swig_incdir_args += '-I' + fs.relative_to(
        abspath,
        meson.project_build_root(),
    )
endforeach

# TODO Add check to ensure we don't miss any
swig_gen_java_source_names = [
    'ActionType.java',
    'BooleanVector.java',
    'CharVector.java',
    'CMMCore.java',
    'Configuration.java',
    'DeviceDetectionStatus.java',
    'DeviceInitializationState.java',
    'DeviceNotification.java',
    'DeviceType.java',
    'DoubleVector.java',
    'FocusDirection.java',
    'LongVector.java',
    'Metadata.java',
    'MetadataArrayTag.java',
    'MetadataError.java',
    'MetadataSingleTag.java',
    'MetadataTag.java',
    'MMCoreJ.java',
    'MMCoreJConstants.java',
    'MMCoreJJNI.java',
    'MMEventCallback.java',
    'pair_ss.java',
    'PortType.java',
    'PropertySetting.java',
    'PropertyType.java',
    'StrMap.java',
    'StrVector.java',
    'SWIGTYPE_p_std__istringstream.java',
    'UnsignedVector.java',
]

swig_gen_java_src_dir = 'generated-java/mmcorej'
swig_gen_java_sources = []
foreach src : swig_gen_java_source_names
    swig_gen_java_sources += swig_gen_java_src_dir / src
endforeach

swig_gen = custom_target(
    'swig-mmcorej',
    input: 'MMCoreJ.i',
    output: [
        'MMCoreJ_swig_wrap.cpp',  # @OUTPUT0@, swig_gen[0]
        'MMCoreJ_swig_wrap.h',    # @OUTPUT1@, swig_gen[1]
        swig_gen_java_source_names,
    ],
    depfile: 'MMCoreJ_swig_wrap.d',
    # Run via wrapper script because outdir needs to pre-exist.
    command: [
        import('python').find_installation(),
        files('scripts/run_swig.py'),
        '--outdir', '@OUTDIR@' / swig_gen_java_src_dir,
        '--',
        swig,
        '-c++',
        '-java',
        '-package', 'mmcorej',
        '-module', 'MMCoreJ',
        swig_incdir_args,
        '-MD', '-MF', '@DEPFILE@',
        '-o', '@OUTPUT0@',
        '-oh', '@OUTPUT1@',
        '-outdir', '@OUTDIR@' / swig_gen_java_src_dir,
        '@INPUT@',
    ],
)

swig_gen_cpp_sources = [swig_gen[0], swig_gen[1]]

# Note that JNI libraries are supposed to be dylibs, not Mach-O bundles, on
# macOS, even though they are loaded by dlopen() (other platforms don't have
# this distinction). So we use shared_library(), not shared_module().
shared_library(
    'MMCoreJ_wrap',
    swig_gen_cpp_sources,
    dependencies: [
        jni_dep,
        mmcore_dep,
        threads_dep,
    ],
    gnu_symbol_visibility: 'inlineshidden',
    install: false,
)

# Note: We do not compile Java sources; leave that to Maven.
