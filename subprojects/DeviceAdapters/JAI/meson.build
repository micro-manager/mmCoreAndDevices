

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'JAI.cpp',
	'PropertyHandlers.cpp'
]


shared_library('JAI', sources, include_directories: includes, dependencies : deps, install: true)

