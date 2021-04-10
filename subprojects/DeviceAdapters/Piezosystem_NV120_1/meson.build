

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Piezosystem_NV120_1.cpp'
]


shared_library('Piezosystem_NV120_1', sources, include_directories: includes, dependencies : deps, install: true)

