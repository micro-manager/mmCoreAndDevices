

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ITKHydra.cpp'
]


shared_library('HydraLMT200', sources, dependencies : deps, install: true)

