

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Cobolt.cpp'
]


shared_library('Cobolt', sources, dependencies : deps, install: true)

