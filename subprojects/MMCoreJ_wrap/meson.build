project('MMCoreJ_wrap', ['cpp'], version : '0.0.1')

add_project_arguments('-D_USRDLL', '-DMMCOREJ_WRAP_EXPORTS', language: 'cpp')

if host_machine.system() == 'windows'
	add_project_arguments('-DWIN32', '-D_WINDOWS', language: ['cpp'])
endif

if get_option('buildtype').startswith('release')
    add_project_arguments('-DNDEBUG', language : ['cpp'])
else
    add_project_arguments('-DDEBUG', language : ['cpp'])
endif


### Copy the pre-written source to the build dir. This also creates the javaSource/main/java/mmcorej path that we need to exist so SWIG can output there without an error.
py_script = '''
import os, shutil
shutil.copytree(os.path.join(r"@1@", "src"),
    os.path.join(r"@0@", "javaSource"), 
    dirs_exist_ok=True)
'''.format(meson.project_build_root(), meson.current_source_dir())

python = import('python').find_installation('python')
cmd = run_command(python, '-c', py_script)
if cmd.returncode() != 0
    error(cmd.stderr())
endif
	
swig = find_program('swig') # Make sure you have swig3 on your path.
cpp_source = custom_target(
    'MMCoreJ_generate_source',
    input: ['MMCoreJ.i'],
    output: ['@BASENAME@.cxx'],  # I'm not sure why but swig doesn't want to have the .h defined here, it'll still be generated.
    command: [swig, '-c++', '-java', '-package', 'mmcorej', '-module', 'MMCoreJ', '-outdir', '@OUTDIR@' / 'javaSource' / 'main' / 'java' / 'mmcorej', 
        '-o', '@OUTPUT@', '@INPUT@']
)

install_subdir(meson.project_build_root() / 'javaSource', install_dir: '.')  # Copy java source from the build directory to install at install time.

deps = [
    dependency('jni'),
	dependency('MMCore', fallback: 'MMCore')
]

includes = [
	include_directories('.')
]

shared_library('MMCoreJ_wrap', cpp_source, include_directories: includes, dependencies: deps, install: true)