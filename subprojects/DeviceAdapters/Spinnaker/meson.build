

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'SpinnakerCamera.cpp'
]


shared_library('Spinnaker', sources, include_directories: includes, dependencies : deps, install: true)

