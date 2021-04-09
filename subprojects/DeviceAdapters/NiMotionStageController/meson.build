

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'NiMotionControl.cpp',
	'ZStage.cpp'
]


shared_library('NiMotionStageController', sources, dependencies : deps, install: true)

