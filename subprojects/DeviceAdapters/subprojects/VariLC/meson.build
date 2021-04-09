project('VariLC', 'cpp', version : '0.0.1')

add_project_arguments('-DMODULE_EXPORTS', language: ['cpp'])

if host_machine.system() == 'windows'
	add_project_arguments('-DWIN32', '-D_WINDOWS', language: ['cpp'])
endif

if get_option('buildtype').startswith('release')
    add_project_arguments('-D_NDEBUG', language : ['cpp'])
else
    add_project_arguments('-D_DEBUG', language : ['cpp'])
endif

includes = include_directories('.') # include the main directory

#MMDev_proj = subproject('MMDevice')
#MMDev_dep = MMDev_proj.get_variable('MMDev_dep')
MMDevice_dep = dependency('MMDevice', fallback: ['MMDevice', 'MMDevice_dep'])

boost_dep = dependency('boost', modules: ['chrono', 'date_time', 'regex', 'system', 'thread'])

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'VariLC.cpp',
]


shared_library('VariLC', sources, include_directories: includes, dependencies : deps)
