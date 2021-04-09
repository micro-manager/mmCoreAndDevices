

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'CoherentScientificRemote.cpp'
]


shared_library('CoherentScientificRemote', sources, dependencies : deps, install: true)

