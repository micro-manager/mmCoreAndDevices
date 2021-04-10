

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'NI100X.cpp'
]


shared_library('NI100X', sources, include_directories: includes, dependencies : deps, install: true)

