

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'ThorMFC1.cpp'
]


shared_library('ThorlabsMFC1', sources, include_directories: includes, dependencies : deps, install: true)

