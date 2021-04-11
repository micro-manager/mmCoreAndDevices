

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'LEDArray.cpp'
]


shared_library('IlluminateLEDArray', sources, include_directories: includes, dependencies : deps, install: true)

