

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Skyra.cpp'
]


shared_library('Skyra', sources, include_directories: includes, dependencies : deps, install: true)

