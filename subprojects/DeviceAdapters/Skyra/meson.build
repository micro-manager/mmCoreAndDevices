

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Skyra.cpp'
]


shared_library('Skyra', sources, dependencies : deps, install: true)

