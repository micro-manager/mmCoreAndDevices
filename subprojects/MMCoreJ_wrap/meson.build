project('MMCoreJ_wrap', 'cpp', version : '0.0.1')

add_project_arguments('-D_USRDLL', '-DMMCOREJ_WRAP_EXPORTS', language: 'cpp')

if host_machine.system() == 'windows'
	add_project_arguments('-DWIN32', '-D_WINDOWS', language: ['cpp'])
endif

if get_option('buildtype').startswith('release')
    add_project_arguments('-DNDEBUG', language : ['cpp'])
else
    add_project_arguments('-DDEBUG', language : ['cpp'])
endif


includes = [
	include_directories('.')
	]
	
swig = find_program('swig') # Make sure you have swig3 on your path.
swig_gen = generator(
    swig,
    output: ['@BASENAME@.cxx'],
    arguments: ['-c++', '-java', '-package', 'mmcorej', '-module', 'MMCoreJ',  '-o', './@OUTPUT@',
                '-outdir', '.', '-I../src', '-MMD',
                '@INPUT@'],
    depfile: '@BASENAME@.d',
)

cpp_source = swig_gen.process('MMCoreJ.i')

deps = [
    dependency('jni'),
	dependency('MMCore', fallback: 'MMCore')
]

shared_library('MMCoreJ_wrap', cpp_source, include_directories: includes, dependencies: deps, install: true)