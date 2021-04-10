

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Piezosystem_30DV50.cpp'
]


shared_library('Piezosystem_30DV50', sources, include_directories: includes, dependencies : deps, install: true)

