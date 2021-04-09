

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CoherentCube.cpp'
]


shared_library('CoherentCube', sources, dependencies : deps, install: true)

