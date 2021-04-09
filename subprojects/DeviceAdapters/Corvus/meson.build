

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Corvus.cpp'
]


shared_library('Corvus', sources, dependencies : deps, install: true)

