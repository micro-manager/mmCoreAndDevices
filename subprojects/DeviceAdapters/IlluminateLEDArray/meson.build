

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LEDArray.cpp'
]


shared_library('IlluminateLEDArray', sources, dependencies : deps, install: true)

