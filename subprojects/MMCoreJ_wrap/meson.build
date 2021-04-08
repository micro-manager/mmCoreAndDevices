project('MMCoreJ_wrap', 'cpp', version : '0.0.1')

add_project_arguments('-DWIN32', '-DDEBUG', '-D_USRDLL', '-DMMCOREJ_WRAP_EXPORTS', '-D_WINDOWS', language: 'cpp')

includes = [
	include_directories('.') # include the projconfig.h file.
	]
	
swig = find_program('swig')
swig_gen = generator(
    swig,
    output: ['@BASENAME@.cxx'],
    arguments: ['-c++', '-java', '-package', 'mmcorej', '-module', 'MMCoreJ',  '-o', './@OUTPUT@',
                '-outdir', '.', '-I../src', '-MMD',
                '@INPUT@'],
    depfile: '@BASENAME@.d',
)

swig_source = 'MMCoreJ.i'



MMCore_proj = subproject('MMCore')
MMCore_dep = MMCore_proj.get_variable('MMCore_dep')

deps = [
    dependency('jni'),
	MMCore_dep,
	#dependency('boost', modules: ['chrono', 'date_time', 'regex', 'system', 'thread'], fallback: 'MMCore')
]

cpp_source = swig_gen.process(swig_source)

shared_library('MMCoreJ_wrap', cpp_source, include_directories: includes, dependencies: deps)