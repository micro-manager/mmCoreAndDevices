

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Sensicam.cpp'
]


shared_library('Sensicam', sources, include_directories: includes, dependencies : deps, install: true)

