

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'RaptorEPIX.cpp'
]


shared_library('RaptorEPIX', sources, include_directories: includes, dependencies : deps, install: true)

