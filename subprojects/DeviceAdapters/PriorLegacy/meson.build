

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PriorLegacy.cpp'
]


shared_library('PriorLegacy', sources, include_directories: includes, dependencies : deps, install: true)

