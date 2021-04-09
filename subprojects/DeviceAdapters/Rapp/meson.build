

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Rapp.cpp'
]


shared_library('Rapp', sources, dependencies : deps, install: true)

