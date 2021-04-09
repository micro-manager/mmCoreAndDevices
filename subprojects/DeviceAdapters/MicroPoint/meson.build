

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'MicroPoint.cpp'
]


shared_library('MicroPoint', sources, dependencies : deps, install: true)

