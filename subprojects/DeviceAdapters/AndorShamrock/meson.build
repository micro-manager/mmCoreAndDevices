
includes = [
    include_directories('.'), # include the main directory
    ]

deps = [
	MMDevice_dep,
	dependency('AndorShamrockSDK', fallback: '3rdPartyPrivate')
]
	
sources = [  # Must explicitly list all source files.
	'AndorShamrock.cpp',
	'ModuleAPIFunctions.cpp',
	'ShamrockConstants.cpp'
]


library('AndorShamrock', sources, include_directories: includes, dependencies : deps, install: true)

