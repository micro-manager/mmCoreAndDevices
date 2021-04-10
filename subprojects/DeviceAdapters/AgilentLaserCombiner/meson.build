

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AgilentLaserCombiner.cpp'
]


shared_library('AgilentLaserCombiner', sources, include_directories: includes, dependencies : deps, install: true)

