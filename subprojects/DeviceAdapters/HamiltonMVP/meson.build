

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'HamiltonMVP.cpp'
]


shared_library('HamiltonMVP', sources, dependencies : deps, install: true)

