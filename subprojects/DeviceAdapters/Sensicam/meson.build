

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Sensicam.cpp'
]


shared_library('Sensicam', sources, dependencies : deps, install: true)

