

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'FLICamera.cpp'
]


shared_library('FLICamera', sources, include_directories: includes, dependencies : deps, install: true)

