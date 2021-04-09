

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'FirstLightImagingCameras.cpp'
]


shared_library('Fli', sources, dependencies : deps, install: true)

