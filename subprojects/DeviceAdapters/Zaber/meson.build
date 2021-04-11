

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'FilterCubeTurret.cpp',
	'FilterWheel.cpp',
	'Illuminator.cpp',
	'Stage.cpp',
	'XYStage.cpp',
	'Zaber.cpp'
]


shared_library('Zaber', sources, include_directories: includes, dependencies : deps, install: true)

