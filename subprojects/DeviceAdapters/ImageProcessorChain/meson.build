

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ImageProcessorChain.cpp'
]


shared_library('ImageProcessorChain', sources, dependencies : deps, install: true)

