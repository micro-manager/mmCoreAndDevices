

includes = [
    include_directories('.'), # include the main directory
    ]

deps = [
	MMDevice_dep,
	dependency('boost'),
	dependency('AndorLaserCombinerSDK', fallback:'3rdPartyPrivate')
	]
	
sources = [  # Must explicitly list all source files.
	'AndorLaserCombiner.cpp'
]


library('AndorLaserCombiner', sources, include_directories: includes, dependencies : deps, install: true)

