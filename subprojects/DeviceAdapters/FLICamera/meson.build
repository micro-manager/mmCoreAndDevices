

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


shared_library('FLICamera', sources, dependencies : deps, install: true)

