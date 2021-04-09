

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Neos.cpp'
]


shared_library('Neos', sources, dependencies : deps, install: true)

