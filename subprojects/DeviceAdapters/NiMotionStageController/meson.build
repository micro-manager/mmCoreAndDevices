

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'NiMotionControl.cpp',
	'ZStage.cpp'
]


library('NiMotionStageController', sources, include_directories: includes, dependencies : deps, install: true)

