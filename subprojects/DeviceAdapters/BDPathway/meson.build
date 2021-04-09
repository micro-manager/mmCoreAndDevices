

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'BDPathway.cpp',
	'PathwayHub.cpp'
]


shared_library('BDPathway', sources, dependencies : deps, install: true)

