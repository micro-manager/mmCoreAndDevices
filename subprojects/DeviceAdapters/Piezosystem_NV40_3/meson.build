

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Piezosystem_NV40_3.cpp'
]


shared_library('Piezosystem_NV40_3', sources, dependencies : deps, install: true)

