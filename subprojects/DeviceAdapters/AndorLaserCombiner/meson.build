

includes = [
    include_directories('.'), # include the main directory
	include_directories(join_paths(MM_3RDPARTY_PRIVATE, 'Andor', 'ALC', 'SDK2.4', 'Include'))
    ]

deps = [
	MMDevice_dep,
	dependency('boost')
	]
	
sources = [  # Must explicitly list all source files.
	'AndorLaserCombiner.cpp'
]


library('AndorLaserCombiner', sources, include_directories: includes, dependencies : deps, install: true)

