

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Piezosystem_NV40_1.cpp'
]


shared_library('Piezosystem_NV40_1', sources, include_directories: includes, dependencies : deps, install: true)

