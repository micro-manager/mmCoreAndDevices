

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'OasisControl.cpp'
]


shared_library('ObjectiveImaging', sources, dependencies : deps, install: true)

