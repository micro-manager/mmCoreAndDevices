

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Utilities.cpp'
]


shared_library('Utilities', sources, include_directories: includes, dependencies : deps, install: true)

