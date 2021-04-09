

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AndorShamrock.cpp',
	'ModuleAPIFunctions.cpp',
	'ShamrockConstants.cpp'
]


shared_library('AndorShamrock', sources, dependencies : deps, install: true)

