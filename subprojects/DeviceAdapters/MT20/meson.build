

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
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


shared_library('MT20', sources, dependencies : deps, install: true)

