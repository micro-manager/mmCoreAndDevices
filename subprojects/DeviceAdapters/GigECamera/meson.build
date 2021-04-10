

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'GigECamera.cpp',
	'GigECameraAcq.cpp',
	'GigECameraAction.cpp',
	'GigENodes.cpp'
]


shared_library('GigECamera', sources, include_directories: includes, dependencies : deps, install: true)

