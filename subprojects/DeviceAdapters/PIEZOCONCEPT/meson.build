

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'PIEZOCONCEPT.cpp'
]


shared_library('PIEZOCONCEPT', sources, dependencies : deps, install: true)

