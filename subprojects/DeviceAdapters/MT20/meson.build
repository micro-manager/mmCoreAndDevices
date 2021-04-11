

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'MT20.cpp',
	'MT20hub.cpp',
	'tinystr.cpp',
	'tinyxml.cpp',
	'tinyxmlerror.cpp',
	'tinyxmlparser.cpp'
]


shared_library('MT20', sources, include_directories: includes, dependencies : deps, install: true)

