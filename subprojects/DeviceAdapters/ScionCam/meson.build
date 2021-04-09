

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'capture.cpp',
	'ScionCamera.cpp',
	'utilities.cpp'
]


shared_library('ScionCam', sources, dependencies : deps, install: true)

