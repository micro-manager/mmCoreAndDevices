

includes = [
    include_directories('.') # include the main directory
    ]

deps = [
	boost_dep,
	MMDevice_dep
	]
	
sources = [  # Must explicitly list all source files.
	'Tofra.cpp'
]


shared_library('Tofra', sources, dependencies : deps, install: true)

