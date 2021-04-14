

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'capture.cpp',
	'ScionCamera.cpp',
	'utilities.cpp'
]


library('ScionCam', sources, include_directories: includes, dependencies : deps, install: true)

