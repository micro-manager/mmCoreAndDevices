

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'SpinnakerCamera.cpp'
]


shared_library('Spinnaker', sources, dependencies : deps, install: true)

