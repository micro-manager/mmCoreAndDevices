

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'MMITC18.cpp'
]


shared_library('ITC18', sources, include_directories: includes, dependencies : deps, install: true)

