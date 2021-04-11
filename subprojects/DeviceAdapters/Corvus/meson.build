

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Corvus.cpp'
]


shared_library('Corvus', sources, include_directories: includes, dependencies : deps, install: true)

