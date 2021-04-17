project('MMCorePy_wrap', 'cpp', version : '0.0.1')

add_project_arguments('-DMMCOREPY_WRAP_EXPORTS', language: 'cpp')

if host_machine.system() == 'windows'
	add_project_arguments('-DWIN32', '-D_WINDOWS', '__WIN32__', language: ['cpp'])
endif

if get_option('buildtype').startswith('release')
    add_project_arguments('-DNDEBUG', language : ['cpp'])
else
    add_project_arguments('-D_DEBUG', language : ['cpp'])
endif

	
swig = find_program('swig')
cpp_source = custom_target(
    'MMCorePy_generate_source',
    input: ['MMCorePy.i'],
    output: ['@BASENAME@.cxx', '@BASENAME@.py'],  # I'm not sure why but swig doesn't want to have the .h defined here, it'll still be generated.
    install: true,
    install_dir: [false, 'MMCorePySource'],
    command: [swig, '-c++', '-python', '-outdir', '@OUTDIR@', 
        '-o', '@OUTPUT0@', '@INPUT@']
)
#install_data(meson.project_build_root() / 'MMCorePy.py', install_dir: 'MMCorePySource')

# Get the numpy include path
pymod = import('python')
python = pymod.find_installation('python2', modules: ['numpy'])
cmd = run_command(python, '-c',  'import numpy; print(numpy.get_include())')
np_inc = include_directories(cmd.stdout().strip())

includes = [
	include_directories('.'),
	np_inc
	]

deps = [
    python.dependency(default_options: ['buildtype=release']),  # Even if we are building debug get the normal linker path, most people don't have the debug dependency installed.
	dependency('MMCore', fallback: 'MMCore')
    ]


shared_library('MMCorePy_wrap', cpp_source, include_directories: includes, dependencies: deps, install: true)