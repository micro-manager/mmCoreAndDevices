
includes = include_directories('.') # include the main directory

boost_dep = dependency('boost', modules: ['chrono', 'date_time', 'regex', 'system', 'thread'])

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'VarispecLCTF.cpp',
]


shared_library('VarispecLCTF', sources, include_directories: includes, dependencies : deps)
