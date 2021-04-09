

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ParallelPort.cpp'
]


shared_library('ParallelPort', sources, dependencies : deps, install: true)

