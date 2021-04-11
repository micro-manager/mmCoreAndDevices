

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AndorShamrock.cpp',
	'ModuleAPIFunctions.cpp',
	'ShamrockConstants.cpp'
]


shared_library('AndorShamrock', sources, include_directories: includes, dependencies : deps, install: true)

