

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AndorLaserCombiner.cpp'
]


shared_library('AndorLaserCombiner', sources, dependencies : deps, install: true)

