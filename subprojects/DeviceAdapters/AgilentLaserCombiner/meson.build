
SDK_dep = dependency('AgilentLaserCombiner', fallback: '3rdPartyPrivate')

includes = [
    include_directories('.'), # include the main directory
    ]

deps = [
	MMDevice_dep,
	SDK_dep
	]
	
sources = [  # Must explicitly list all source files.
	'AgilentLaserCombiner.cpp'
]


library('AgilentLaserCombiner', sources, include_directories: includes, dependencies : deps, install: true)

