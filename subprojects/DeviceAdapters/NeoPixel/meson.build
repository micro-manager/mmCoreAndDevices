

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'NeoPixel.cpp'
]


shared_library('NeoPixel', sources, include_directories: includes, dependencies : deps, install: true)

