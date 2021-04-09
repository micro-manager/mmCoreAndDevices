

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ArduinoDueFuncGen.cpp'
]


shared_library('ArduinoDueFuncGen', sources, dependencies : deps, install: true)

