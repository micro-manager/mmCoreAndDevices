

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'OasisControl.cpp'
]


shared_library('ObjectiveImaging', sources, include_directories: includes, dependencies : deps, install: true)

