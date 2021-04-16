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
swig_gen = generator(
    swig,
    output: ['@BASENAME@.cxx'],
    arguments: ['-c++', '-python', '-o', './@OUTPUT@',
                '-outdir', '.', '-I../src', '-MMD',
                '@INPUT@'],
    depfile: '@BASENAME@.d',
)

swig_source = 'MMCorePy.i'

MMCore_dep = dependency('MMCore', fallback: 'MMCore')

pymod = import('python')
python = pymod.find_installation('python2', modules: ['numpy'])
cmd = run_command(python, '-c',  'import numpy; print(numpy.get_include())')
np_inc = include_directories(cmd.stdout().strip())

includes = [
	include_directories('.'),
	np_inc
	]

deps = [
    python.dependency(),
	MMCore_dep
    ]

cpp_source = swig_gen.process(swig_source)


shared_library('MMCorePy_wrap', cpp_source, include_directories: includes, dependencies: deps, install: true)