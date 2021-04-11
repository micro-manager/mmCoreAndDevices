

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Rapp.cpp'
]


shared_library('Rapp', sources, include_directories: includes, dependencies : deps, install: true)

