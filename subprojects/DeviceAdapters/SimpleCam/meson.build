

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CameraFrontend.cpp'
]


shared_library('SimpleCam', sources, dependencies : deps, install: true)

