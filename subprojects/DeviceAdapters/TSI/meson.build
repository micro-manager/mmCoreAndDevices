

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ColorProcess.cpp',
	'PolarizationProcess.cpp',
	'TSI3Cam.cpp',
	'Tsi3CamPropertyHandlers.cpp',
	'TSICam.cpp',
	'TsiCamPropertyHandlers.cpp',
	'TSILibrary.cpp'
]


shared_library('TSI', sources, include_directories: includes, dependencies : deps, install: true)

