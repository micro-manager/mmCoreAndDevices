project('MMCoreJ_wrap', 'cpp', version : '0.0.1')


conf = configuration_data({
	'WIN32': true,
	'DEBUG': true,
	'_LIB': true,
	'_WINDOWS': true
	})
	
configure_file( output : 'projconfig.h',
               configuration : conf)
includes = [
	include_directories('.') # include the config.h file.
	]
	
swig = find_program('swig')
swig_gen = generator(
    swig,
    output: ['@BASENAME@.cxx'],
    arguments: ['-c++', '-python', '-py3', '-o', './@OUTPUT@',
                '-outdir', '.', '-I../src', '-MMD',
                '@INPUT@'],
    depfile: '@BASENAME@.d',
)

swig_sources = [
    'MMCoreJ_wrap.i',
]

wrapper_deps = [
    dependency('java'),
]

foreach swig_source: swig_sources
    cpp_source = swig_gen.process(swig_source)
endforeach

shared_library('MMCoreJ_wrap', 'MMCore.cpp', include_directories: includes)