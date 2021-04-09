

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
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


shared_library('Zaber', sources, dependencies : deps, install: true)

